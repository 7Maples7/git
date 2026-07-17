#include "RadarThreeRecognizerBridge.h"

#include <QCoreApplication>
#include <QDebug>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStringList>

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);
    const QStringList args = app.arguments();
    if(args.size() < 4)
    {
        qCritical().noquote()
            << "Usage: radar_three_cls_client <checkpoint.pth> <radar_three_recognizer_worker.py|.exe> <dcls_echo_blob.bin> [python] [device]";
        return 2;
    }

    QFile blobFile(args.at(3));
    if(!blobFile.open(QIODevice::ReadOnly))
    {
        qCritical().noquote() << "failed to read echo blob:" << args.at(3);
        return 2;
    }
    const QByteArray blob = blobFile.readAll();

    RadarThreeCls::RadarThreeRecognizerInitCfg cfg;
    cfg.checkpointPath = args.at(1);
    cfg.workerScriptPath = args.at(2);
    if(args.size() >= 5)
    {
        cfg.pythonExecutable = args.at(4);
    }
    if(args.size() >= 6)
    {
        cfg.device = args.at(5);
    }

    RadarThreeCls::RadarThreeRecognizerBridge bridge;
    if(!bridge.init(cfg))
    {
        qCritical().noquote() << "init failed:" << bridge.lastError();
        return 1;
    }

    const RadarThreeCls::RadarThreeRecognizeResult result = bridge.recognize(blob);
    QJsonObject out = result.rawData;
    out.insert("ok", result.ok);
    out.insert("errorCode", result.errorCode);
    out.insert("errorMsg", result.errorMsg);
    qInfo().noquote() << QJsonDocument(out).toJson(QJsonDocument::Indented);

    bridge.shutdown();
    return result.ok ? 0 : 1;
}
