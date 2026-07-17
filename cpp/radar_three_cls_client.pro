QT += core
CONFIG += console c++11
CONFIG -= app_bundle

TEMPLATE = app
TARGET = radar_three_cls_client

SOURCES += \
    RadarThreeRecognizerBridge.cpp \
    example_main.cpp

HEADERS += \
    RadarThreeRecognizerBridge.h
