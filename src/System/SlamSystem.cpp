/**
* This file is part of LSD-SLAM.
*
* Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
* For more information see <http://vision.in.tum.de/lsdslam> 
*
* LSD-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* LSD-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "SlamSystem.hpp"

#include "DataStructures/Frame.hpp"
#include "Tracking/SE3Tracker.hpp"
#include "Tracking/Sim3Tracker.hpp"
#include "DepthEstimation/DepthMap.hpp"
#include "Tracking/TrackingReference.hpp"
#include "LiveSLAMWrapper.hpp"
#include "Util/globalFuncs.hpp"
#include "GlobalMapping/KeyframeGraph.hpp"
#include "GlobalMapping/TrackableKeyframeSearch.hpp"
#include "GlobalMapping/g2oTypeSim3Sophus.hpp"
#include "IOWrapper/ImageDisplay.hpp"
#include "IOWrapper/Output3DWrapper.hpp"
#include <g2o/core/robust_kernel_impl.h>
#include "DataStructures/FrameMemory.hpp"
#include <deque>
#include "CameraModel/ProjCameraModel.hpp"

// for mkdir
#include <sys/types.h>
#include <sys/stat.h>

#ifdef ANDROID
#include <android/log.h>
#endif

#include "opencv2/opencv.hpp"

using namespace lsd_slam;


SlamSystem::SlamSystem(const CameraModel &model,
	bool enableLoopClosure, bool singleThread,
	DepthMapInitMode depthMapInitMode,
	bool saveTrackingInfo)
	: model(model.clone()),
	loopClosureEnabled(enableLoopClosure),
	plotTracking(false),
	singleThread(singleThread),
	singleThreadMappingInterval(1),
	relocalizer(model),
	saveTrackingInfo(saveTrackingInfo)
{
	if(model.w%16 != 0 || model.h%16!=0)
	{
		printf("image dimensions must be multiples of 16! Please crop your "
			"images / video accordingly.\n");
		assert(false);
	}

	trackingIsGood = true;

	if (saveTrackingInfo) {
		saveTrackingInfoStream.open(resourcesDir() + "/TrackInfo.txt",
			std::ios::out | std::ios::trunc);
	}

	currentKeyframe =  nullptr;
	trackingReferenceFrameSharedPT = nullptr;
	keyframeGraph = new KeyframeGraph();
	createNewKeyframe = false;

	map =  new DepthMap(model, depthMapInitMode);
	
	
	newConstraintAdded = false;
	haveUnmergedOptimizationOffset = false;


	tracker = new SE3Tracker(model);
	// Do not use more than 4 levels for odometry tracking
	for (int level = 4; level < PYRAMID_LEVELS; ++level)
		tracker->settings.maxItsPerLvl[level] = 0;
	trackingKeyframe = new TrackingKeyframe();
	mappingTrackingReference = new TrackingKeyframe();


	if(loopClosureEnabled)
	{
		trackableKeyframeSearch = new TrackableKeyframeSearch(keyframeGraph,model);
		constraintTracker = new Sim3Tracker(model);
		constraintSE3Tracker = new SE3Tracker(model);
		newKFTrackingReference = new TrackingKeyframe();
		candidateTrackingReference = new TrackingKeyframe();
	}
	else
	{
		constraintSE3Tracker = 0;
		trackableKeyframeSearch = 0;
		constraintTracker = 0;
		newKFTrackingReference = 0;
		candidateTrackingReference = 0;
	}


	outputWrapper = 0;

	keepRunning = true;
	doFinalOptimization = false;
	depthMapScreenshotFlag = false;
	lastTrackingClosenessScore = 0;

	if (!singleThread) {
		thread_mapping = boost::thread(&SlamSystem::mappingThreadLoop, this);
	}

	if(loopClosureEnabled)
	{
		if (!singleThread) {
			thread_constraint_search = boost::thread(&SlamSystem::constraintSearchThreadLoop, this);
			thread_optimization = boost::thread(&SlamSystem::optimizationThreadLoop, this);
		}
	}



	msTrackFrame = msOptimizationIteration = msFindConstraintsItaration = msFindReferences = 0;
	nTrackFrame = nOptimizationIteration = nFindConstraintsItaration = nFindReferences = 0;
	nAvgTrackFrame = nAvgOptimizationIteration = nAvgFindConstraintsItaration = nAvgFindReferences = 0;
	gettimeofday(&lastHzUpdate, NULL);

}

SlamSystem::~SlamSystem()
{
	keepRunning = false;

	// make sure none is waiting for something.
	printf("... waiting for SlamSystem's threads to exit\n");
	newFrameMappedSignal.notify_all();
	unmappedTrackedFramesSignal.notify_all();
	newKeyframeCreatedSignal.notify_all();
	newConstraintCreatedSignal.notify_all();

	thread_mapping.join();
	thread_constraint_search.join();
	thread_optimization.join();
	printf("DONE waiting for SlamSystem's threads to exit\n");

	if(trackableKeyframeSearch != 0) delete trackableKeyframeSearch;
	if(constraintTracker != 0) delete constraintTracker;
	if(constraintSE3Tracker != 0) delete constraintSE3Tracker;
	if(newKFTrackingReference != 0) delete newKFTrackingReference;
	if(candidateTrackingReference != 0) delete candidateTrackingReference;

	delete mappingTrackingReference;
	delete map;
	delete trackingKeyframe;
	delete tracker;

	// make shure to reset all shared pointers to all frames before deleting the keyframegraph!
	unmappedTrackedFrames.clear();
	latestFrameTriedForReloc.reset();
	latestTrackedFrame.reset();
	currentKeyframe.reset();
	trackingReferenceFrameSharedPT.reset();

	// delte keyframe graph
	delete keyframeGraph;

	FrameMemory::getInstance().releaseBuffes();


	Util::closeAllWindows();
	
}

void SlamSystem::setVisualization(Output3DWrapper* outputWrapper)
{
	this->outputWrapper = outputWrapper;
}

void SlamSystem::mergeOptimizationOffset()
{
	// update all vertices that are in the graph!
	poseConsistencyMutex.lock();

	bool needPublish = false;
	if(haveUnmergedOptimizationOffset)
	{
		keyframeGraph->keyframesAllMutex.lock_shared();
		for(unsigned int i=0;i<keyframeGraph->keyframesAll.size(); i++)
			keyframeGraph->keyframesAll[i]->pose->applyPoseGraphOptResult();
		keyframeGraph->keyframesAllMutex.unlock_shared();

		haveUnmergedOptimizationOffset = false;
		needPublish = true;
	}

	poseConsistencyMutex.unlock();






	if(needPublish)
		publishKeyframeGraph();
}



void SlamSystem::mappingThreadLoop()
{
	printf("Started mapping thread!\n");
	while(keepRunning)
	{
		mappingThreadLoopIteration();
	}
	printf("Exited mapping thread \n");
}

void SlamSystem::mappingThreadLoopIteration()
{
	if (!doMappingIteration())
	{
		boost::unique_lock<boost::mutex> lock(unmappedTrackedFramesMutex);
		unmappedTrackedFramesSignal.timed_wait(lock,boost::posix_time::milliseconds(200));	// slight chance of deadlock otherwise
		lock.unlock();
	}

	newFrameMappedMutex.lock();
	newFrameMappedSignal.notify_all();
	newFrameMappedMutex.unlock();
}

void SlamSystem::finalize()
{
	printf("Finalizing Graph... finding final constraints!!\n");

	lastNumConstraintsAddedOnFullRetrack = 1;
	while(lastNumConstraintsAddedOnFullRetrack != 0)
	{
		doFullReConstraintTrack = true;
		usleep(200000);
	}


	printf("Finalizing Graph... optimizing!!\n");
	doFinalOptimization = true;
	newConstraintMutex.lock();
	newConstraintAdded = true;
	newConstraintCreatedSignal.notify_all();
	newConstraintMutex.unlock();
	while(doFinalOptimization)
	{
		usleep(200000);
	}


	printf("Finalizing Graph... publishing!!\n");
	unmappedTrackedFramesMutex.lock();
	unmappedTrackedFramesSignal.notify_one();
	unmappedTrackedFramesMutex.unlock();
	while(doFinalOptimization)
	{
		usleep(200000);
	}
	boost::unique_lock<boost::mutex> lock(newFrameMappedMutex);
	newFrameMappedSignal.wait(lock);
	newFrameMappedSignal.wait(lock);

	usleep(200000);
	printf("Done Finalizing Graph.!!\n");
}


void SlamSystem::constraintSearchThreadLoop()
{
	printf("Started  constraint search thread!\n");

	while(keepRunning){
		constraintSearchThreadLoopIteration();
	}

	printf("Exited constraint search thread \n");
}

void SlamSystem::constraintSearchThreadLoopIteration()
{
	static boost::unique_lock<boost::mutex> lock(newKeyframeMutex);
	static int failedToRetrack = 0;
	if(newKeyframes.size() == 0)
	{
		lock.unlock();
		keyframeGraph->keyframesForRetrackMutex.lock();
		bool doneSomething = false;
		if(keyframeGraph->keyframesForRetrack.size() > 10)
		{
			std::deque< Frame* >::iterator toReTrack = keyframeGraph->keyframesForRetrack.begin() + (rand() % (keyframeGraph->keyframesForRetrack.size()/3));
			Frame* toReTrackFrame = *toReTrack;

			keyframeGraph->keyframesForRetrack.erase(toReTrack);
			keyframeGraph->keyframesForRetrack.push_back(toReTrackFrame);

			keyframeGraph->keyframesForRetrackMutex.unlock();

			int found = findConstraintsForNewKeyframes(toReTrackFrame, false, false, 2.0);
			if(found == 0)
				failedToRetrack++;
			else
				failedToRetrack=0;

			if(failedToRetrack < (int)keyframeGraph->keyframesForRetrack.size() - 5)
				doneSomething = true;
		}
		else
			keyframeGraph->keyframesForRetrackMutex.unlock();

		lock.lock();

		if(!doneSomething)
		{
			if(enablePrintDebugInfo && printConstraintSearchInfo)
				printf("nothing to re-track... waiting.\n");
			newKeyframeCreatedSignal.timed_wait(lock,boost::posix_time::milliseconds(500));

		}
	}
	else
	{
		Frame* newKF = newKeyframes.front();
		newKeyframes.pop_front();
		lock.unlock();

		struct timeval tv_start, tv_end;
		gettimeofday(&tv_start, NULL);

		findConstraintsForNewKeyframes(newKF, true, true, 1.0);
		failedToRetrack=0;
		gettimeofday(&tv_end, NULL);
		msFindConstraintsItaration = 0.9f*msFindConstraintsItaration + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
		nFindConstraintsItaration++;

		FrameMemory::getInstance().pruneActiveFrames();
		lock.lock();
	}


	if(doFullReConstraintTrack)
	{
		lock.unlock();
		printf("Optizing Full Map!\n");

		int added = 0;
		for(unsigned int i=0;i<keyframeGraph->keyframesAll.size();i++)
		{
			if(keyframeGraph->keyframesAll[i]->pose->isInGraph)
				added += findConstraintsForNewKeyframes(keyframeGraph->keyframesAll[i], false, false, 1.0);
		}

		printf("Done optizing Full Map! Added %d constraints.\n", added);

		doFullReConstraintTrack = false;

		lastNumConstraintsAddedOnFullRetrack = added;
		lock.lock();
	}
}

void SlamSystem::optimizationThreadLoop()
{
	printf("Started optimization thread \n");

	while(keepRunning){
		optimizationThreadLoopIteration();
	}

	printf("Exited optimization thread \n");
}

void SlamSystem::optimizationThreadLoopIteration()
{
	boost::unique_lock<boost::mutex> lock(newConstraintMutex);
	if(!newConstraintAdded)
		newConstraintCreatedSignal.timed_wait(lock,boost::posix_time::milliseconds(2000));	// slight chance of deadlock otherwise
	newConstraintAdded = false;
	lock.unlock();

	if(doFinalOptimization)
	{
		printf("doing final optimization iteration!\n");
		optimizationIteration(50, 0.001f);
		doFinalOptimization = false;
	}
	while(optimizationIteration(5, 0.02f));
}

void SlamSystem::publishKeyframeGraph()
{
	if (outputWrapper != nullptr)
		outputWrapper->publishKeyframeGraph(keyframeGraph);
}

void SlamSystem::requestDepthMapScreenshot(const std::string& filename)
{
	depthMapScreenshotFilename = filename;
	depthMapScreenshotFlag = true;
}

void SlamSystem::finishCurrentKeyframe()
{
	if(enablePrintDebugInfo && printThreadingInfo)
		printf("FINALIZING KF %d\n", currentKeyframe->id());

	map->finalizeKeyframe();

	if(loopClosureEnabled)
	{
		mappingTrackingReference->importFrame(currentKeyframe.get());
		currentKeyframe->setPermaRef(mappingTrackingReference);
		mappingTrackingReference->invalidate();

		if(currentKeyframe->idxInKeyframes < 0)
		{
			keyframeGraph->keyframesAllMutex.lock();
			currentKeyframe->idxInKeyframes = keyframeGraph->keyframesAll.size();
			keyframeGraph->keyframesAll.push_back(currentKeyframe.get());
			keyframeGraph->totalPoints += currentKeyframe->numPoints;
			keyframeGraph->totalVertices ++;
			keyframeGraph->keyframesAllMutex.unlock();

			if (singleThread) {
				newKeyframes.push_back(currentKeyframe.get());
				constraintSearchThreadLoopIteration();
				optimizationThreadLoopIteration();
			} else {
				newKeyframeMutex.lock();
				newKeyframes.push_back(currentKeyframe.get());
				newKeyframeCreatedSignal.notify_all();
				newKeyframeMutex.unlock();
			}
		}
	}

	if(outputWrapper != 0)
		outputWrapper->publishKeyframe(currentKeyframe.get());
}

void SlamSystem::discardCurrentKeyframe()
{
	if(enablePrintDebugInfo && printThreadingInfo)
		printf("DISCARDING KF %d\n", currentKeyframe->id());

	if(currentKeyframe->idxInKeyframes >= 0)
	{
		printf("WARNING: trying to discard a KF that has already been added to the graph... finalizing instead.\n");
		finishCurrentKeyframe();
		return;
	}


	map->invalidate();

	keyframeGraph->allFramePosesMutex.lock();
	for(FramePoseStruct* p : keyframeGraph->allFramePoses)
	{
		if(p->trackingParent != 0 && p->trackingParent->frameID == currentKeyframe->id())
			p->trackingParent = 0;
	}
	keyframeGraph->allFramePosesMutex.unlock();


	keyframeGraph->idToKeyframeMutex.lock();
	keyframeGraph->idToKeyframe.erase(currentKeyframe->id());
	keyframeGraph->idToKeyframeMutex.unlock();

}

void SlamSystem::createNewCurrentKeyframe(std::shared_ptr<Frame> newKeyframeCandidate)
{
	if(enablePrintDebugInfo && printThreadingInfo)
		printf("CREATE NEW KF %d from %d\n", newKeyframeCandidate->id(), currentKeyframe->id());


	if(loopClosureEnabled)
	{
		// add NEW keyframe to id-lookup
		keyframeGraph->idToKeyframeMutex.lock();
		keyframeGraph->idToKeyframe.insert(std::make_pair(newKeyframeCandidate->id(), newKeyframeCandidate));
		keyframeGraph->idToKeyframeMutex.unlock();
	}

	// propagate & make new.
	map->createKeyframe(newKeyframeCandidate.get());

	if(printPropagationStatistics)
	{

		Eigen::Matrix<float, 20, 1> data;
		data.setZero();
		data[0] = runningStats.num_prop_attempts / ((float)model->w*model->h);
		data[1] = (runningStats.num_prop_created + runningStats.num_prop_merged) / (float)runningStats.num_prop_attempts;
		data[2] = runningStats.num_prop_removed_colorDiff / (float)runningStats.num_prop_attempts;

		outputWrapper->publishDebugInfo(data);
	}

	currentKeyframeMutex.lock();
	currentKeyframe = newKeyframeCandidate;
	currentKeyframeMutex.unlock();
}
void SlamSystem::loadNewCurrentKeyframe(Frame* keyframeToLoad)
{
	if(enablePrintDebugInfo && printThreadingInfo)
		printf("RE-ACTIVATE KF %d\n", keyframeToLoad->id());

	map->setFromExistingKF(keyframeToLoad);

	if(enablePrintDebugInfo && printRegularizeStatistics)
		printf("re-activate frame %d!\n", keyframeToLoad->id());

	currentKeyframeMutex.lock();
	currentKeyframe = keyframeGraph->idToKeyframe.find(keyframeToLoad->id())->second;
	currentKeyframe->depthHasBeenUpdatedFlag = false;
	currentKeyframeMutex.unlock();
}

void SlamSystem::changeKeyframe(bool noCreate, bool force, float maxScore)
{
	Frame* newReferenceKF=0;
	std::shared_ptr<Frame> newKeyframeCandidate = latestTrackedFrame;
	if(doKFReActivation && loopClosureEnabled)
	{
		struct timeval tv_start, tv_end;
		gettimeofday(&tv_start, NULL);
		newReferenceKF = trackableKeyframeSearch->findRePositionCandidate(newKeyframeCandidate.get(), maxScore);
		gettimeofday(&tv_end, NULL);
		msFindReferences = 0.9f*msFindReferences + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
		nFindReferences++;
	}

	if(newReferenceKF != 0)
		loadNewCurrentKeyframe(newReferenceKF);
	else
	{
		if(force)
		{
			if(noCreate)
			{
				trackingIsGood = false;
				nextRelocIdx = -1;
				printf("mapping is disabled & moved outside of known map. Starting Relocalizer!\n");
			}
			else
				createNewCurrentKeyframe(newKeyframeCandidate);
		}
	}


	createNewKeyframe = false;
}

bool SlamSystem::updateKeyframe()
{
	std::shared_ptr<Frame> reference = nullptr;
	std::deque< std::shared_ptr<Frame> > references;

	unmappedTrackedFramesMutex.lock();

	// remove frames that have a different tracking parent.
	while(unmappedTrackedFrames.size() > 0 &&
			(!unmappedTrackedFrames.front()->hasTrackingParent() ||
					unmappedTrackedFrames.front()->getTrackingParent() != currentKeyframe.get()))
	{
		unmappedTrackedFrames.front()->clear_refPixelWasGood();
		unmappedTrackedFrames.pop_front();
	}

	// clone list
	if(unmappedTrackedFrames.size() > 0)
	{
		for(unsigned int i=0;i<unmappedTrackedFrames.size(); i++)
			references.push_back(unmappedTrackedFrames[i]);

		std::shared_ptr<Frame> popped = unmappedTrackedFrames.front();
		unmappedTrackedFrames.pop_front();
		unmappedTrackedFramesMutex.unlock();

		if(enablePrintDebugInfo && printThreadingInfo)
			printf("MAPPING %d on %d to %d (%d frames)\n", currentKeyframe->id(), references.front()->id(), references.back()->id(), (int)references.size());

		map->updateKeyframe(references);

		popped->clear_refPixelWasGood();
		references.clear();
	}
	else
	{
		unmappedTrackedFramesMutex.unlock();
		return false;
	}


	if(enablePrintDebugInfo && printRegularizeStatistics)
	{
		Eigen::Matrix<float, 20, 1> data;
		data.setZero();
		data[0] = static_cast<float>(runningStats.num_reg_created);
		data[2] = static_cast<float>(runningStats.num_reg_smeared);
		data[3] = static_cast<float>(runningStats.num_reg_deleted_secondary);
		data[4] = static_cast<float>(runningStats.num_reg_deleted_occluded);
		data[5] = static_cast<float>(runningStats.num_reg_blacklisted);

		data[6] = static_cast<float>(runningStats.num_observe_created);
		data[7] = static_cast<float>(runningStats.num_observe_create_attempted);
		data[8] = static_cast<float>(runningStats.num_observe_updated);
		data[9] = static_cast<float>(runningStats.num_observe_update_attempted);


		data[10] = static_cast<float>(runningStats.num_observe_good);
		data[11] = static_cast<float>(runningStats.num_observe_inconsistent);
		data[12] = static_cast<float>(runningStats.num_observe_notfound);
		data[13] = static_cast<float>(runningStats.num_observe_skip_oob);
		data[14] = static_cast<float>(runningStats.num_observe_skip_fail);

		outputWrapper->publishDebugInfo(data);
	}



	if(outputWrapper != 0 && continuousPCOutput && currentKeyframe != 0)
		outputWrapper->publishKeyframe(currentKeyframe.get());

	return true;
}


void SlamSystem::addTimingSamples()
{
	map->addTimingSample();
	struct timeval now;
	gettimeofday(&now, NULL);
	float sPassed = ((now.tv_sec-lastHzUpdate.tv_sec) + (now.tv_usec-lastHzUpdate.tv_usec)/1000000.0f);
	if(sPassed > 1.0f)
	{
		nAvgTrackFrame = 0.8f*nAvgTrackFrame + 0.2f*(nTrackFrame / sPassed); nTrackFrame = 0;
		nAvgOptimizationIteration = 0.8f*nAvgOptimizationIteration + 0.2f*(nOptimizationIteration / sPassed); nOptimizationIteration = 0;
		nAvgFindReferences = 0.8f*nAvgFindReferences + 0.2f*(nFindReferences / sPassed); nFindReferences = 0;

		if(trackableKeyframeSearch != 0)
		{
			trackableKeyframeSearch->nAvgTrackPermaRef = 0.8f*trackableKeyframeSearch->nAvgTrackPermaRef + 0.2f*(trackableKeyframeSearch->nTrackPermaRef / sPassed); trackableKeyframeSearch->nTrackPermaRef = 0;
		}
		nAvgFindConstraintsItaration = 0.8f*nAvgFindConstraintsItaration + 0.2f*(nFindConstraintsItaration / sPassed); nFindConstraintsItaration = 0;
		nAvgOptimizationIteration = 0.8f*nAvgOptimizationIteration + 0.2f*(nOptimizationIteration / sPassed); nOptimizationIteration = 0;

		lastHzUpdate = now;


		if(enablePrintDebugInfo && printOverallTiming)
		{
			printf("MapIt: %3.1fms (%.1fHz); Track: %3.1fms (%.1fHz); Create: %3.1fms (%.1fHz); FindRef: %3.1fms (%.1fHz); PermaTrk: %3.1fms (%.1fHz); Opt: %3.1fms (%.1fHz); FindConst: %3.1fms (%.1fHz);\n",
					map->msUpdate, map->nAvgUpdate,
					msTrackFrame, nAvgTrackFrame,
					map->msCreate+map->msFinalize, map->nAvgCreate,
					msFindReferences, nAvgFindReferences,
					trackableKeyframeSearch != 0 ? trackableKeyframeSearch->msTrackPermaRef : 0, trackableKeyframeSearch != 0 ? trackableKeyframeSearch->nAvgTrackPermaRef : 0,
					msOptimizationIteration, nAvgOptimizationIteration,
					msFindConstraintsItaration, nAvgFindConstraintsItaration);
		}
	}

}


void SlamSystem::debugDisplayDepthMap()
{
	map->debugPlotDepthMap();
	double scale = 1;
	if(currentKeyframe != 0 && currentKeyframe != 0)
		scale = currentKeyframe->getScaledCamToWorld().scale();
	// debug plot depthmap
	char buf1[200];
	char buf2[200];


	snprintf(buf1,200,"Map: Upd %3.0fms (%2.0fHz); Trk %3.0fms (%2.0fHz); %d / %d / %d",
			map->msUpdate, map->nAvgUpdate,
			msTrackFrame, nAvgTrackFrame,
			currentKeyframe->numFramesTrackedOnThis, currentKeyframe->numMappedOnThis, (int)unmappedTrackedFrames.size());

	snprintf(buf2,200,"dens %2.0f%%; good %2.0f%%; scale %2.2f; res %2.1f/; usg %2.0f%%; Map: %d F, %d KF, %d E, %.1fm Pts",
			100*currentKeyframe->numPoints/(float)(model->w*model->h),
			100*tracking_lastGoodPerBad,
			scale,
			tracking_lastResidual,
			100*tracking_lastUsage,
			(int)keyframeGraph->allFramePoses.size(),
			keyframeGraph->totalVertices,
			(int)keyframeGraph->edgesAll.size(),
			1e-6 * (float)keyframeGraph->totalPoints);


	if(onSceenInfoDisplay)
		printMessageOnCVImage(map->debugImageDepth, buf1, buf2);
	if (displayDepthMap)
		Util::displayImage( "DebugWindow DEPTH", map->debugImageDepth, false );

	int pressedKey = Util::waitKey(1);
	handleKey(pressedKey);
}


void SlamSystem::takeRelocalizeResult()
{
	Frame* keyframe;
	int succFrameID;
	SE3 succFrameToKF_init;
	std::shared_ptr<Frame> succFrame;
	relocalizer.stop();
	relocalizer.getResult(keyframe, succFrame, succFrameID, succFrameToKF_init);
	assert(keyframe != 0);

	loadNewCurrentKeyframe(keyframe);

	currentKeyframeMutex.lock();
	trackingKeyframe->importFrame(currentKeyframe.get());
	trackingReferenceFrameSharedPT = currentKeyframe;
	currentKeyframeMutex.unlock();

	tracker->trackFrame(
			trackingKeyframe,
			succFrame.get(),
			succFrameToKF_init);

	if(!tracker->trackingWasGood || tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount) < 1-0.75f*(1-MIN_GOODPERGOODBAD_PIXEL))
	{
		if(enablePrintDebugInfo && printRelocalizationInfo)
			printf("RELOCALIZATION FAILED BADLY! discarding result.\n");
		trackingKeyframe->invalidate();
	}
	else
	{
		keyframeGraph->addFrame(succFrame.get());

		unmappedTrackedFramesMutex.lock();
		if(unmappedTrackedFrames.size() < 50)
			unmappedTrackedFrames.push_back(succFrame);
		unmappedTrackedFramesMutex.unlock();

		currentKeyframeMutex.lock();
		createNewKeyframe = false;
		trackingIsGood = true;
		currentKeyframeMutex.unlock();
	}
}

bool SlamSystem::doMappingIteration()
{
	if(currentKeyframe == 0)
		return false;

	if(!doMapping && currentKeyframe->idxInKeyframes < 0)
	{
		if(currentKeyframe->numMappedOnThisTotal >= MIN_NUM_MAPPED)
			finishCurrentKeyframe();
		else
			discardCurrentKeyframe();

		map->invalidate();
		printf("Finished KF %d as Mapping got disabled!\n",currentKeyframe->id());

		changeKeyframe(true, true, 1.0f);
	}

	mergeOptimizationOffset();
	addTimingSamples();

	if(dumpMap)
	{
		keyframeGraph->dumpMap(packagePath+"/save");
		dumpMap = false;
	}


	// set mappingFrame
	if(trackingIsGood)
	{
		if(!doMapping)
		{
			//printf("tryToChange refframe, lastScore %f!\n", lastTrackingClosenessScore);
			if(lastTrackingClosenessScore > 1)
				changeKeyframe(true, false, lastTrackingClosenessScore * 0.75f);

			if (displayDepthMap || depthMapScreenshotFlag)
				debugDisplayDepthMap();

			return false;
		}


		if (createNewKeyframe)
		{
			finishCurrentKeyframe();
			changeKeyframe(false, true, 1.0f);


			if (displayDepthMap || depthMapScreenshotFlag)
				debugDisplayDepthMap();
		}
		else
		{
			bool didSomething = updateKeyframe();

			if (displayDepthMap || depthMapScreenshotFlag)
				debugDisplayDepthMap();
			if(!didSomething)
				return false;
		}

		return true;
	}
	else
	{
		// invalidate map if it was valid.
		if(map->isValid())
		{
			if(currentKeyframe->numMappedOnThisTotal >= MIN_NUM_MAPPED)
				finishCurrentKeyframe();
			else
				discardCurrentKeyframe();

			map->invalidate();
		}

		// start relocalizer if it isnt running already
		if(!relocalizer.isRunning)
			relocalizer.start(keyframeGraph->keyframesAll);

		// did we find a frame to relocalize with?
		if(relocalizer.waitResult(50))
			takeRelocalizeResult();


		return true;
	}
}


void SlamSystem::gtDepthInit(uchar* image, float* depth, double timeStamp, int id)
{
	printf("Doing GT initialization!\n");

	currentKeyframeMutex.lock();

	currentKeyframe.reset(new Frame(id, *model, timeStamp, image));
	currentKeyframe->setDepthFromGroundTruth(depth);

	map->initializeFromGTDepth(currentKeyframe.get());
	keyframeGraph->addFrame(currentKeyframe.get());

	currentKeyframeMutex.unlock();

	if(doSlam)
	{
		keyframeGraph->idToKeyframeMutex.lock();
		keyframeGraph->idToKeyframe.insert(std::make_pair(currentKeyframe->id(), currentKeyframe));
		keyframeGraph->idToKeyframeMutex.unlock();
	}
	if(continuousPCOutput && outputWrapper != 0) outputWrapper->publishKeyframe(currentKeyframe.get());

	printf("Done GT initialization!\n");
}


void SlamSystem::randomInit(uchar* image, double timeStamp, int id)
{
	std::cout << "Doing Random initialization!" << std::endl;

	if(!doMapping) {
		std::cout << "WARNING: mapping is disabled, but we just initialized... "
			"THIS WILL NOT WORK! Set doMapping to true." << std::endl;
		throw std::runtime_error("Mapping disabled when initialising.");
	}


	currentKeyframeMutex.lock();

	currentKeyframe.reset(new Frame(id, *model, timeStamp, image));
	map->initializeRandomly(currentKeyframe.get());
	keyframeGraph->addFrame(currentKeyframe.get());

	currentKeyframeMutex.unlock();

	if(doSlam)
	{
		keyframeGraph->idToKeyframeMutex.lock();
		keyframeGraph->idToKeyframe.insert(std::make_pair(currentKeyframe->id(), currentKeyframe));
		keyframeGraph->idToKeyframeMutex.unlock();
	}
	if(continuousPCOutput && outputWrapper != 0) outputWrapper->publishKeyframe(currentKeyframe.get());


	if (displayDepthMap || depthMapScreenshotFlag)
		debugDisplayDepthMap();


	printf("Done Random initialization!\n");

}

void SlamSystem::trackFrame(
	uchar* image, unsigned int frameID, bool blockUntilMapped, double timestamp)
{
	// Create new frame
	std::shared_ptr<Frame> trackingNewFrame(new Frame(frameID, *model, timestamp, image));

	if (!trackingIsGood)
	{
		relocalizer.updateCurrentFrame(trackingNewFrame);

		unmappedTrackedFramesMutex.lock();
		unmappedTrackedFramesSignal.notify_one();
		unmappedTrackedFramesMutex.unlock();
		return;
	}

	currentKeyframeMutex.lock();
	bool my_createNewKeyframe = createNewKeyframe;	// pre-save here, to make decision afterwards.
	if (trackingKeyframe->keyframe != currentKeyframe.get() ||
		currentKeyframe->depthHasBeenUpdatedFlag)
	{
		trackingKeyframe->importFrame(currentKeyframe.get());
		currentKeyframe->depthHasBeenUpdatedFlag = false;
		trackingReferenceFrameSharedPT = currentKeyframe;
	}

	FramePoseStruct* trackingReferencePose = trackingKeyframe->keyframe->pose;
	currentKeyframeMutex.unlock();

	// DO TRACKING & Show tracking result.
	if (enablePrintDebugInfo && printThreadingInfo)
		printf("TRACKING %d on %d\n", trackingNewFrame->id(),
		trackingReferencePose->frameID);


	poseConsistencyMutex.lock_shared();
	SE3 frameToReference_initialEstimate = se3FromSim3(
		trackingReferencePose->getCamToWorld().inverse() *
		keyframeGraph->allFramePoses.back()->getCamToWorld());
	poseConsistencyMutex.unlock_shared();

	struct timeval tv_start, tv_end;
	gettimeofday(&tv_start, NULL);

	SE3 newRefToFrame_poseUpdate = tracker->trackFrame(
		trackingKeyframe,
		trackingNewFrame.get(),
		frameToReference_initialEstimate);
	
	if (saveTrackingInfo) {
		saveTrackingInfoStream <<
			"**Tracking frame " << frameID << "**\n" <<
			newRefToFrame_poseUpdate << "\n\n" << std::endl;
	}


	gettimeofday(&tv_end, NULL);
	msTrackFrame = 0.9f*msTrackFrame + 0.1f*((tv_end.tv_sec - tv_start.tv_sec)*1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
	nTrackFrame++;

	tracking_lastResidual = tracker->lastResidual;
	tracking_lastUsage = tracker->pointUsage;
	tracking_lastGoodPerBad = tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount);
	tracking_lastGoodPerTotal = tracker->lastGoodCount / (trackingNewFrame->width(SE3TRACKING_MIN_LEVEL)*trackingNewFrame->height(SE3TRACKING_MIN_LEVEL));


	if (manualTrackingLossIndicated || tracker->diverged || (keyframeGraph->keyframesAll.size() > INITIALIZATION_PHASE_COUNT && !tracker->trackingWasGood))
	{
		printf("TRACKING LOST for frame %d (%1.2f%% good Points, which is %1.2f%% of available points, %s)!\n",
			trackingNewFrame->id(),
			100 * tracking_lastGoodPerTotal,
			100 * tracking_lastGoodPerBad,
			tracker->diverged ? "DIVERGED" : "NOT DIVERGED");

		trackingKeyframe->invalidate();

		trackingIsGood = false;
		nextRelocIdx = -1;

		unmappedTrackedFramesMutex.lock();
		unmappedTrackedFramesSignal.notify_one();
		unmappedTrackedFramesMutex.unlock();

		manualTrackingLossIndicated = false;
		return;
	}



	if (plotTracking)
	{
		Eigen::Matrix<float, 20, 1> data;
		data.setZero();
		data[0] = tracker->lastResidual;

		data[3] = tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount);
		data[4] = 4 * tracker->lastGoodCount / (model->w*model->h);
		data[5] = tracker->pointUsage;

		data[6] = tracker->affineEstimation_a;
		data[7] = tracker->affineEstimation_b;
		outputWrapper->publishDebugInfo(data);
	}

	keyframeGraph->addFrame(trackingNewFrame.get());


	//Sim3 lastTrackedCamToWorld = mostCurrentTrackedFrame->getScaledCamToWorld();//  mostCurrentTrackedFrame->TrackingParent->getScaledCamToWorld() * sim3FromSE3(mostCurrentTrackedFrame->thisToParent_SE3TrackingResult, 1.0);
	if (outputWrapper != 0)
	{
		outputWrapper->publishTrackedFrame(trackingNewFrame.get());
	}


	// Keyframe selection
	latestTrackedFrame = trackingNewFrame;
	if (!my_createNewKeyframe && currentKeyframe->numMappedOnThisTotal > MIN_NUM_MAPPED)
	{
		Sophus::Vector3d dist = newRefToFrame_poseUpdate.translation() * currentKeyframe->meanIdepth;
		float minVal = fmin(0.2f + keyframeGraph->keyframesAll.size() * 0.8f / INITIALIZATION_PHASE_COUNT, 1.0f);

		if (keyframeGraph->keyframesAll.size() < INITIALIZATION_PHASE_COUNT)	minVal *= 0.7f;

		lastTrackingClosenessScore = trackableKeyframeSearch->getRefFrameScore(float(dist.dot(dist)), tracker->pointUsage);

		if (lastTrackingClosenessScore > minVal)
		{
			createNewKeyframe = true;

			if (printKeyframeSelectionInfo)
				printf("SELECT %d on %d! dist %.3f + usage %.3f = %.3f > 1\n",
					trackingNewFrame->id(), trackingNewFrame->getTrackingParent()->id(),
					float(dist.dot(dist)), tracker->pointUsage,
					trackableKeyframeSearch->getRefFrameScore(float(dist.dot(dist)), tracker->pointUsage));
		}
		else
		{
			if (printKeyframeSelectionInfo && printNonKeyframeInfo)
				printf("SKIPPD %d on %d! dist %.3f + usage %.3f = %.3f > 1\n",
					trackingNewFrame->id(), trackingNewFrame->getTrackingParent()->id(),
					float(dist.dot(dist)), tracker->pointUsage,
					trackableKeyframeSearch->getRefFrameScore(float(dist.dot(dist)),
						tracker->pointUsage));

		}
	}


	unmappedTrackedFramesMutex.lock();
	if (unmappedTrackedFrames.size() < 50 || (unmappedTrackedFrames.size() < 100 && trackingNewFrame->getTrackingParent()->numMappedOnThisTotal < 10))
		unmappedTrackedFrames.push_back(trackingNewFrame);
	unmappedTrackedFramesSignal.notify_one();
	unmappedTrackedFramesMutex.unlock();

	// implement blocking
	if (blockUntilMapped && trackingIsGood)
	{
		boost::unique_lock<boost::mutex> lock(newFrameMappedMutex);
		while (unmappedTrackedFrames.size() > 0)
		{
			printf("TRACKING IS BLOCKING, waiting for %d frames to finish mapping.\n", (int)unmappedTrackedFrames.size());
			newFrameMappedSignal.wait(lock);
		}
		lock.unlock();
	}

	//In single threaded mode, run mapping, constraint search and optimisation
	// in the same thread as tracking.
	if (singleThread) {
		static unsigned long counter = 0;

		if (counter % singleThreadMappingInterval == 0) {
			mappingThreadLoopIteration();
		}
		++counter;
	}
}


float SlamSystem::tryTrackSim3(
		TrackingKeyframe* A, TrackingKeyframe* B,
		int lvlStart, int lvlEnd,
		bool useSSE,
		Sim3 &AtoB, Sim3 &BtoA,
		KFConstraintStruct* e1, KFConstraintStruct* e2 )
{
	BtoA = constraintTracker->trackFrameSim3(
			A,
			B->keyframe,
			BtoA,
			lvlStart,lvlEnd);
	Matrix7x7 BtoAInfo = constraintTracker->lastSim3Hessian;
	float BtoA_meanResidual = constraintTracker->lastResidual;
	float BtoA_meanDResidual = constraintTracker->lastDepthResidual;
	float BtoA_meanPResidual = constraintTracker->lastPhotometricResidual;
	float BtoA_usage = constraintTracker->pointUsage;


	if (constraintTracker->diverged ||
		BtoA.scale() > 1 / Sophus::SophusConstants<sophusType>::epsilon() ||
		BtoA.scale() < Sophus::SophusConstants<sophusType>::epsilon() ||
		BtoAInfo(0,0) == 0 ||
		BtoAInfo(6,6) == 0)
	{
		return 1e20f;
	}


	AtoB = constraintTracker->trackFrameSim3(
			B,
			A->keyframe,
			AtoB,
			lvlStart,lvlEnd);
	Matrix7x7 AtoBInfo = constraintTracker->lastSim3Hessian;
	float AtoB_meanResidual = constraintTracker->lastResidual;
	float AtoB_meanDResidual = constraintTracker->lastDepthResidual;
	float AtoB_meanPResidual = constraintTracker->lastPhotometricResidual;
	float AtoB_usage = constraintTracker->pointUsage;


	if (constraintTracker->diverged ||
		AtoB.scale() > 1 / Sophus::SophusConstants<sophusType>::epsilon() ||
		AtoB.scale() < Sophus::SophusConstants<sophusType>::epsilon() ||
		AtoBInfo(0,0) == 0 ||
		AtoBInfo(6,6) == 0)
	{
		return 1e20f;
	}

	// Propagate uncertainty (with d(a * b) / d(b) = Adj_a) and calculate Mahalanobis norm
	Matrix7x7 datimesb_db = AtoB.cast<float>().Adj();
	Matrix7x7 diffHesse = (AtoBInfo.inverse() + datimesb_db * BtoAInfo.inverse() * datimesb_db.transpose()).inverse();
	Vector7 diff = (AtoB * BtoA).log().cast<float>();


	float reciprocalConsistency = (diffHesse * diff).dot(diff);


	if(e1 != 0 && e2 != 0)
	{
		e1->firstFrame = A->keyframe;
		e1->secondFrame = B->keyframe;
		e1->secondToFirst = BtoA;
		e1->information = BtoAInfo.cast<double>();
		e1->meanResidual = BtoA_meanResidual;
		e1->meanResidualD = BtoA_meanDResidual;
		e1->meanResidualP = BtoA_meanPResidual;
		e1->usage = BtoA_usage;

		e2->firstFrame = B->keyframe;
		e2->secondFrame = A->keyframe;
		e2->secondToFirst = AtoB;
		e2->information = AtoBInfo.cast<double>();
		e2->meanResidual = AtoB_meanResidual;
		e2->meanResidualD = AtoB_meanDResidual;
		e2->meanResidualP = AtoB_meanPResidual;
		e2->usage = AtoB_usage;

		e1->reciprocalConsistency = e2->reciprocalConsistency = reciprocalConsistency;
	}

	return reciprocalConsistency;
}


void SlamSystem::testConstraint(
		Frame* candidate,
		KFConstraintStruct* &e1_out, KFConstraintStruct* &e2_out,
		const Sim3 &candidateToFrame_initialEstimate,
		float strictness)
{
	candidateTrackingReference->importFrame(candidate);

	Sim3 FtoC = candidateToFrame_initialEstimate.inverse(), CtoF = candidateToFrame_initialEstimate;
	Matrix7x7 FtoCInfo, CtoFInfo;

	float err_level3 = tryTrackSim3(
			newKFTrackingReference, candidateTrackingReference,	// A = frame; b = candidate
			SIM3TRACKING_MAX_LEVEL-1, 3,
			USESSE,
			FtoC, CtoF);

	if(err_level3 > 3000*strictness)
	{
		if(enablePrintDebugInfo && printConstraintSearchInfo)
			printf("FAILE %d -> %d (lvl %d): errs (%.1f / - / -).",
				newKFTrackingReference->frameID, candidateTrackingReference->frameID,
				3,
				sqrtf(err_level3));

		e1_out = e2_out = 0;

		newKFTrackingReference->keyframe->trackingFailed.insert(std::pair<Frame*,Sim3>(candidate, candidateToFrame_initialEstimate));
		return;
	}

	float err_level2 = tryTrackSim3(
			newKFTrackingReference, candidateTrackingReference,	// A = frame; b = candidate
			2, 2,
			USESSE,
			FtoC, CtoF);

	if(err_level2 > 4000*strictness)
	{
		if(enablePrintDebugInfo && printConstraintSearchInfo)
			printf("FAILE %d -> %d (lvl %d): errs (%.1f / %.1f / -).",
				newKFTrackingReference->frameID, candidateTrackingReference->frameID,
				2,
				sqrtf(err_level3), sqrtf(err_level2));

		e1_out = e2_out = 0;
		newKFTrackingReference->keyframe->trackingFailed.insert(std::pair<Frame*,Sim3>(candidate, candidateToFrame_initialEstimate));
		return;
	}

	e1_out = new KFConstraintStruct();
	e2_out = new KFConstraintStruct();


	float err_level1 = tryTrackSim3(
			newKFTrackingReference, candidateTrackingReference,	// A = frame; b = candidate
			1, 1,
			USESSE,
			FtoC, CtoF, e1_out, e2_out);

	if(err_level1 > 6000*strictness)
	{
		if(enablePrintDebugInfo && printConstraintSearchInfo)
			printf("FAILE %d -> %d (lvl %d): errs (%.1f / %.1f / %.1f).",
					newKFTrackingReference->frameID, candidateTrackingReference->frameID,
					1,
					sqrtf(err_level3), sqrtf(err_level2), sqrtf(err_level1));

		delete e1_out;
		delete e2_out;
		e1_out = e2_out = 0;
		newKFTrackingReference->keyframe->trackingFailed.insert(std::pair<Frame*,Sim3>(candidate, candidateToFrame_initialEstimate));
		return;
	}


	if(enablePrintDebugInfo && printConstraintSearchInfo)
		printf("ADDED %d -> %d: errs (%.1f / %.1f / %.1f).",
			newKFTrackingReference->frameID, candidateTrackingReference->frameID,
			sqrtf(err_level3), sqrtf(err_level2), sqrtf(err_level1));


	const float kernelDelta = 5 * sqrt(6000*loopclosureStrictness);
	e1_out->robustKernel = new g2o::RobustKernelHuber();
	e1_out->robustKernel->setDelta(kernelDelta);
	e2_out->robustKernel = new g2o::RobustKernelHuber();
	e2_out->robustKernel->setDelta(kernelDelta);
}

int SlamSystem::findConstraintsForNewKeyframes(Frame* newKeyframe, bool forceParent, bool useFABMAP, float closeCandidatesTH)
{
	if(!newKeyframe->hasTrackingParent())
	{
		newConstraintMutex.lock();
		keyframeGraph->addKeyframe(newKeyframe);
		newConstraintAdded = true;
		newConstraintCreatedSignal.notify_all();
		newConstraintMutex.unlock();
		return 0;
	}

	if(!forceParent && (newKeyframe->lastConstraintTrackedCamToWorld * newKeyframe->getScaledCamToWorld().inverse()).log().norm() < 0.01)
		return 0;


	newKeyframe->lastConstraintTrackedCamToWorld = newKeyframe->getScaledCamToWorld();

	// =============== get all potential candidates and their initial relative pose. =================
	std::vector<KFConstraintStruct*> constraints;
	Frame* fabMapResult = 0;
	std::unordered_set<Frame*, std::hash<Frame*>, std::equal_to<Frame*>,
		Eigen::aligned_allocator< Frame* > > candidates = trackableKeyframeSearch->findCandidates(newKeyframe, fabMapResult, useFABMAP, closeCandidatesTH != 0.f);
	std::map< Frame*, Sim3, std::less<Frame*>, Eigen::aligned_allocator<std::pair<Frame*, Sim3> > > candidateToFrame_initialEstimateMap;


	// erase the ones that are already neighbours.
	for(std::unordered_set<Frame*>::iterator c = candidates.begin(); c != candidates.end();)
	{
		if(newKeyframe->neighbors.find(*c) != newKeyframe->neighbors.end())
		{
			if(enablePrintDebugInfo && printConstraintSearchInfo)
				printf("SKIPPING %d on %d cause it already exists as constraint.\n", (*c)->id(), newKeyframe->id());
			c = candidates.erase(c);
		}
		else
			++c;
	}

	poseConsistencyMutex.lock_shared();
	for (Frame* candidate : candidates)
	{
		Sim3 candidateToFrame_initialEstimate = newKeyframe->getScaledCamToWorld().inverse() * candidate->getScaledCamToWorld();
		candidateToFrame_initialEstimateMap[candidate] = candidateToFrame_initialEstimate;
	}

	std::unordered_map<Frame*, int> distancesToNewKeyframe;
	if(newKeyframe->hasTrackingParent())
		keyframeGraph->calculateGraphDistancesToFrame(newKeyframe->getTrackingParent(), &distancesToNewKeyframe);
	poseConsistencyMutex.unlock_shared();





	// =============== distinguish between close and "far" candidates in Graph =================
	// Do a first check on trackability of close candidates.
	std::unordered_set<Frame*, std::hash<Frame*>, std::equal_to<Frame*>,
		Eigen::aligned_allocator< Frame* > > closeCandidates;
	std::vector<Frame*> farCandidates;
	Frame* parent = newKeyframe->hasTrackingParent() ? newKeyframe->getTrackingParent() : 0;

	int closeFailed = 0;
	int closeInconsistent = 0;

	SO3 disturbance = SO3::exp(Sophus::Vector3d(0.05,0,0));

	for (Frame* candidate : candidates)
	{
		if (candidate->id() == newKeyframe->id())
			continue;
		if(!candidate->pose->isInGraph)
			continue;
		if(newKeyframe->hasTrackingParent() && candidate == newKeyframe->getTrackingParent())
			continue;
		if(candidate->idxInKeyframes < INITIALIZATION_PHASE_COUNT)
			continue;

		SE3 c2f_init = se3FromSim3(candidateToFrame_initialEstimateMap[candidate].inverse()).inverse();
		c2f_init.so3() = c2f_init.so3() * disturbance;
		SE3 c2f = constraintSE3Tracker->trackFrameOnPermaref(candidate, newKeyframe, c2f_init);
		if(!constraintSE3Tracker->trackingWasGood) {closeFailed++; continue;}


		SE3 f2c_init = se3FromSim3(candidateToFrame_initialEstimateMap[candidate]).inverse();
		f2c_init.so3() = disturbance * f2c_init.so3();
		SE3 f2c = constraintSE3Tracker->trackFrameOnPermaref(newKeyframe, candidate, f2c_init);
		if(!constraintSE3Tracker->trackingWasGood) {closeFailed++; continue;}

		if((f2c.so3() * c2f.so3()).log().norm() >= 0.09) {closeInconsistent++; continue;}

		closeCandidates.insert(candidate);
	}



	int farFailed = 0;
	int farInconsistent = 0;
	for (Frame* candidate : candidates)
	{
		if (candidate->id() == newKeyframe->id())
			continue;
		if(!candidate->pose->isInGraph)
			continue;
		if(newKeyframe->hasTrackingParent() && candidate == newKeyframe->getTrackingParent())
			continue;
		if(candidate->idxInKeyframes < INITIALIZATION_PHASE_COUNT)
			continue;

		if(candidate == fabMapResult)
		{
			farCandidates.push_back(candidate);
			continue;
		}

		if(distancesToNewKeyframe.at(candidate) < 4)
			continue;

		farCandidates.push_back(candidate);
	}




	size_t closeAll = closeCandidates.size();
	size_t farAll = farCandidates.size();

	// erase the ones that we tried already before (close)
	for(std::unordered_set<Frame*>::iterator c = closeCandidates.begin(); c != closeCandidates.end();)
	{
		if(newKeyframe->trackingFailed.find(*c) == newKeyframe->trackingFailed.end())
		{
			++c;
			continue;
		}
		auto range = newKeyframe->trackingFailed.equal_range(*c);

		bool skip = false;
		Sim3 f2c = candidateToFrame_initialEstimateMap[*c].inverse();
		for (auto it = range.first; it != range.second; ++it)
		{
			if((f2c * it->second).log().norm() < 0.1)
			{
				skip=true;
				break;
			}
		}

		if(skip)
		{
			if(enablePrintDebugInfo && printConstraintSearchInfo)
				printf("SKIPPING %d on %d (NEAR), cause we already have tried it.\n", (*c)->id(), newKeyframe->id());
			c = closeCandidates.erase(c);
		}
		else
			++c;
	}

	// erase the ones that are already neighbours (far)
	for(unsigned int i=0;i<farCandidates.size();i++)
	{
		if(newKeyframe->trackingFailed.find(farCandidates[i]) == newKeyframe->trackingFailed.end())
			continue;

		auto range = newKeyframe->trackingFailed.equal_range(farCandidates[i]);

		bool skip = false;
		for (auto it = range.first; it != range.second; ++it)
		{
			if((it->second).log().norm() < 0.2)
			{
				skip=true;
				break;
			}
		}

		if(skip)
		{
			if(enablePrintDebugInfo && printConstraintSearchInfo)
				printf("SKIPPING %d on %d (FAR), cause we already have tried it.\n", farCandidates[i]->id(), newKeyframe->id());
			farCandidates[i] = farCandidates.back();
			farCandidates.pop_back();
			i--;
		}
	}



	if (enablePrintDebugInfo && printConstraintSearchInfo)
		printf("Final Loop-Closure Candidates: %d / %d close (%d failed, %d inconsistent) + %d / %d far (%d failed, %d inconsistent) = %d\n",
				(int)closeCandidates.size(),closeAll, closeFailed, closeInconsistent,
				(int)farCandidates.size(), farAll, farFailed, farInconsistent,
				(int)closeCandidates.size() + (int)farCandidates.size());



	// =============== limit number of close candidates ===============
	// while too many, remove the one with the highest connectivity.
	while((int)closeCandidates.size() > maxLoopClosureCandidates)
	{
		Frame* worst = 0;
		int worstNeighbours = 0;
		for(Frame* f : closeCandidates)
		{
			int neightboursInCandidates = 0;
			for(Frame* n : f->neighbors)
				if(closeCandidates.find(n) != closeCandidates.end())
					neightboursInCandidates++;

			if(neightboursInCandidates > worstNeighbours || worst == 0)
			{
				worst = f;
				worstNeighbours = neightboursInCandidates;
			}
		}

		closeCandidates.erase(worst);
	}


	// =============== limit number of far candidates ===============
	// delete randomly
	int maxNumFarCandidates = (maxLoopClosureCandidates +1) / 2;
	if(maxNumFarCandidates < 5) maxNumFarCandidates = 5;
	while((int)farCandidates.size() > maxNumFarCandidates)
	{
		int toDelete = rand() % farCandidates.size();
		if(farCandidates[toDelete] != fabMapResult)
		{
			farCandidates[toDelete] = farCandidates.back();
			farCandidates.pop_back();
		}
	}







	// =============== TRACK! ===============

	// make tracking reference for newKeyframe.
	newKFTrackingReference->importFrame(newKeyframe);


	for (Frame* candidate : closeCandidates)
	{
		KFConstraintStruct* e1=0;
		KFConstraintStruct* e2=0;

		testConstraint(
				candidate, e1, e2,
				candidateToFrame_initialEstimateMap[candidate],
				loopclosureStrictness);

		if(enablePrintDebugInfo && printConstraintSearchInfo)
			printf(" CLOSE (%d)\n", distancesToNewKeyframe.at(candidate));

		if(e1 != 0)
		{
			constraints.push_back(e1);
			constraints.push_back(e2);

			// delete from far candidates if it's in there.
			for(unsigned int k=0;k<farCandidates.size();k++)
			{
				if(farCandidates[k] == candidate)
				{
					if(enablePrintDebugInfo && printConstraintSearchInfo)
						printf(" DELETED %d from far, as close was successful!\n", candidate->id());

					farCandidates[k] = farCandidates.back();
					farCandidates.pop_back();
				}
			}
		}
	}


	for (Frame* candidate : farCandidates)
	{
		KFConstraintStruct* e1=0;
		KFConstraintStruct* e2=0;

		testConstraint(
				candidate, e1, e2,
				Sim3(),
				loopclosureStrictness);

		if(enablePrintDebugInfo && printConstraintSearchInfo)
			printf(" FAR (%d)\n", distancesToNewKeyframe.at(candidate));

		if(e1 != 0)
		{
			constraints.push_back(e1);
			constraints.push_back(e2);
		}
	}



	if(parent != 0 && forceParent)
	{
		KFConstraintStruct* e1=0;
		KFConstraintStruct* e2=0;
		testConstraint(
				parent, e1, e2,
				candidateToFrame_initialEstimateMap[parent],
				100);
		if(enablePrintDebugInfo && printConstraintSearchInfo)
			printf(" PARENT (0)\n");

		if(e1 != 0)
		{
			constraints.push_back(e1);
			constraints.push_back(e2);
		}
		else
		{
			float downweightFac = 5;
			const float kernelDelta = 5 * sqrt(6000*loopclosureStrictness) / downweightFac;
			printf("warning: reciprocal tracking on new frame failed badly, added odometry edge (Hacky).\n");

			poseConsistencyMutex.lock_shared();
			constraints.push_back(new KFConstraintStruct());
			(constraints.back())->firstFrame = newKeyframe;
			constraints.back()->secondFrame = newKeyframe->getTrackingParent();
			constraints.back()->secondToFirst = constraints.back()->firstFrame->getScaledCamToWorld().inverse() * constraints.back()->secondFrame->getScaledCamToWorld();
			constraints.back()->information  <<
					0.8098,-0.1507,-0.0557, 0.1211, 0.7657, 0.0120, 0,
					-0.1507, 2.1724,-0.1103,-1.9279,-0.1182, 0.1943, 0,
					-0.0557,-0.1103, 0.2643,-0.0021,-0.0657,-0.0028, 0.0304,
					 0.1211,-1.9279,-0.0021, 2.3110, 0.1039,-0.0934, 0.0005,
					 0.7657,-0.1182,-0.0657, 0.1039, 1.0545, 0.0743,-0.0028,
					 0.0120, 0.1943,-0.0028,-0.0934, 0.0743, 0.4511, 0,
					0,0, 0.0304, 0.0005,-0.0028, 0, 0.0228;
			constraints.back()->information *= (1e9/(downweightFac*downweightFac));

			constraints.back()->robustKernel = new g2o::RobustKernelHuber();
			constraints.back()->robustKernel->setDelta(kernelDelta);

			constraints.back()->meanResidual = 10;
			constraints.back()->meanResidualD = 10;
			constraints.back()->meanResidualP = 10;
			constraints.back()->usage = 0;

			poseConsistencyMutex.unlock_shared();
		}
	}


	newConstraintMutex.lock();

	keyframeGraph->addKeyframe(newKeyframe);
	for(unsigned int i=0;i<constraints.size();i++)
		keyframeGraph->insertConstraint(constraints[i]);


	newConstraintAdded = true;
	newConstraintCreatedSignal.notify_all();
	newConstraintMutex.unlock();

	newKFTrackingReference->invalidate();
	candidateTrackingReference->invalidate();



	return constraints.size();
}




bool SlamSystem::optimizationIteration(int itsPerTry, float minChange)
{
	struct timeval tv_start, tv_end;
	gettimeofday(&tv_start, NULL);



	g2oGraphAccessMutex.lock();

	// lock new elements buffer & take them over.
	newConstraintMutex.lock();
	keyframeGraph->addElementsFromBuffer();
	newConstraintMutex.unlock();


	// Do the optimization. This can take quite some time!
	int its = keyframeGraph->optimize(itsPerTry);
	

	// save the optimization result.
	poseConsistencyMutex.lock_shared();
	keyframeGraph->keyframesAllMutex.lock_shared();
	float maxChange = 0;
	float sumChange = 0;
	float sum = 0;
	for(size_t i=0;i<keyframeGraph->keyframesAll.size(); i++)
	{
		// set edge error sum to zero
		keyframeGraph->keyframesAll[i]->edgeErrorSum = 0;
		keyframeGraph->keyframesAll[i]->edgesNum = 0;

		if(!keyframeGraph->keyframesAll[i]->pose->isInGraph) continue;



		// get change from last optimization
		Sim3 a = keyframeGraph->keyframesAll[i]->pose->graphVertex->estimate();
		Sim3 b = keyframeGraph->keyframesAll[i]->getScaledCamToWorld();
		Sophus::Vector7f diff = (a*b.inverse()).log().cast<float>();


		for(int j=0;j<7;j++)
		{
			float d = fabsf((float)(diff[j]));
			if(d > maxChange) maxChange = d;
			sumChange += d;
		}
		sum +=7;

		// set change
		keyframeGraph->keyframesAll[i]->pose->setPoseGraphOptResult(
				keyframeGraph->keyframesAll[i]->pose->graphVertex->estimate());

		// add error
		for(auto edge : keyframeGraph->keyframesAll[i]->pose->graphVertex->edges())
		{
			keyframeGraph->keyframesAll[i]->edgeErrorSum += float(((EdgeSim3*)(edge))->chi2());
			keyframeGraph->keyframesAll[i]->edgesNum++;
		}
	}

	haveUnmergedOptimizationOffset = true;
	keyframeGraph->keyframesAllMutex.unlock_shared();
	poseConsistencyMutex.unlock_shared();

	g2oGraphAccessMutex.unlock();

	if(enablePrintDebugInfo && printOptimizationInfo)
		printf("did %d optimization iterations. Max Pose Parameter Change: %f; avgChange: %f. %s\n", its, maxChange, sumChange / sum,
				maxChange > minChange && its == itsPerTry ? "continue optimizing":"Waiting for addition to graph.");


	gettimeofday(&tv_end, NULL);
	msOptimizationIteration = 0.9f*msOptimizationIteration + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nOptimizationIteration++;


	return maxChange > minChange && its == itsPerTry;
}

void SlamSystem::optimizeGraph()
{
	boost::unique_lock<boost::mutex> g2oLock(g2oGraphAccessMutex);
	keyframeGraph->optimize(1000);
	g2oLock.unlock();
	mergeOptimizationOffset();
}


SE3 SlamSystem::getCurrentPoseEstimate()
{
	SE3 camToWorld = SE3();
	keyframeGraph->allFramePosesMutex.lock_shared();
	if(keyframeGraph->allFramePoses.size() > 0)
		camToWorld = se3FromSim3(keyframeGraph->allFramePoses.back()->getCamToWorld());
	keyframeGraph->allFramePosesMutex.unlock_shared();
	return camToWorld;
}

std::vector<FramePoseStruct*> SlamSystem::getAllPoses()
{
	return keyframeGraph->allFramePoses;
}

DepthMapDebugSettings & lsd_slam::SlamSystem::depthMapDebugSettings()
{
	return map->settings;
}
