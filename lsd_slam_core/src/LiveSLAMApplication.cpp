#include "LiveSLAMApplication.hpp"

LiveSLAMApplication::LiveSLAMApplication(int &argc, char **argv)
	:QApplication(argc, argv)
{}

LiveSLAMApplication::~LiveSLAMApplication()
{}

bool LiveSLAMApplication::notify(QObject *object, QEvent *event)
{
	return QApplication::notify(object, event);
}
