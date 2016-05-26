#pragma once

#include <qapplication.h>

class LiveSLAMApplication : public QApplication
{
public:
	LiveSLAMApplication(int &argc, char **argv);
	virtual ~LiveSLAMApplication();
	bool notify(QObject *object, QEvent *event);
};
