#ifndef RADARTHREERECOGNIZERBRIDGE_H
#define RADARTHREERECOGNIZERBRIDGE_H

#include <QByteArray>
#include <QJsonObject>
#include <QMutex>
#include <QString>

class QProcess;

namespace RadarThreeCls
{
    struct RadarThreeRecognizerInitCfg
    {
        QString pythonExecutable = "python";
        QString workerScriptPath = "";
        QString checkpointPath = "";
        QString device = "auto";
        bool strictModelLoad = true;
        int startupTimeoutMs = 10000;
        int requestTimeoutMs = 5000;
    };

    struct RadarThreeRecognizeResult
    {
        bool ok = false;
        QString errorCode = "";
        QString errorMsg = "";

        qint64 predTargetTypeDec = -1;
        QString predTargetTypeHex = "-1";
        qint64 predClassIdDec = -1;
        QString predClassIdHex = "-1";
        QString predLabel = "";

        double score = 0.0;
        double top1Score = 0.0;
        double top2Score = 0.0;
        double margin = 0.0;

        QJsonObject probabilities;
        QJsonObject probabilitiesByClassId;
        QJsonObject rawData;
    };

    class RadarThreeRecognizerBridge
    {
    public:
        RadarThreeRecognizerBridge();
        ~RadarThreeRecognizerBridge();

        RadarThreeRecognizerBridge(const RadarThreeRecognizerBridge&) = delete;
        RadarThreeRecognizerBridge& operator=(const RadarThreeRecognizerBridge&) = delete;

        bool init(const RadarThreeRecognizerInitCfg& cfg);
        RadarThreeRecognizeResult recognize(const QByteArray& dclsEchoBlob);
        void shutdown();

        bool isReady() const;
        QString lastError() const;

    private:
        bool sendCommandLocked(const QString& cmd, const QJsonObject& payload, QJsonObject* outData);
        bool startWorkerLocked(const RadarThreeRecognizerInitCfg& cfg);
        void stopWorkerLocked();
        QString resolveWorkerEntryPath(const RadarThreeRecognizerInitCfg& cfg) const;
        static bool isWorkerExecutablePath(const QString& path);
        static QString categoryToHex(qint64 value);

    private:
        mutable QMutex mMutex;
        QProcess* mProcess = nullptr;
        int mRequestSeq = 0;
        int mRequestTimeoutMs = 5000;
        bool mReady = false;
        QString mLastError = "";
    };
}

#endif // RADARTHREERECOGNIZERBRIDGE_H
