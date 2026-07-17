#include "RadarThreeRecognizerBridge.h"

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QJsonValue>
#include <QMutexLocker>
#include <QProcess>
#include <QProcessEnvironment>
#include <QStringList>

namespace
{
    static qint64 jsonToInt64(const QJsonObject& obj, const QString& key, qint64 defaultVal = 0)
    {
        if(!obj.contains(key))
        {
            return defaultVal;
        }
        const QJsonValue v = obj.value(key);
        if(v.isDouble())
        {
            return static_cast<qint64>(v.toDouble(defaultVal));
        }
        if(v.isString())
        {
            bool ok = false;
            const qint64 parsed = v.toString().toLongLong(&ok, 0);
            return ok ? parsed : defaultVal;
        }
        return defaultVal;
    }

    static double jsonToDouble(const QJsonObject& obj, const QString& key, double defaultVal = 0.0)
    {
        if(!obj.contains(key))
        {
            return defaultVal;
        }
        const QJsonValue v = obj.value(key);
        if(v.isDouble())
        {
            return v.toDouble(defaultVal);
        }
        if(v.isString())
        {
            bool ok = false;
            const double parsed = v.toString().toDouble(&ok);
            return ok ? parsed : defaultVal;
        }
        return defaultVal;
    }

    static QString jsonToString(const QJsonObject& obj, const QString& key, const QString& defaultVal = "")
    {
        if(!obj.contains(key))
        {
            return defaultVal;
        }
        const QJsonValue v = obj.value(key);
        if(v.isString())
        {
            return v.toString(defaultVal);
        }
        if(v.isDouble())
        {
            return QString::number(v.toDouble(), 'g', 16);
        }
        if(v.isBool())
        {
            return v.toBool() ? "true" : "false";
        }
        return defaultVal;
    }
}

namespace RadarThreeCls
{
    RadarThreeRecognizerBridge::RadarThreeRecognizerBridge()
    {
    }

    RadarThreeRecognizerBridge::~RadarThreeRecognizerBridge()
    {
        this->shutdown();
    }

    bool RadarThreeRecognizerBridge::init(const RadarThreeRecognizerInitCfg& cfg)
    {
        QMutexLocker locker(&this->mMutex);
        this->mLastError.clear();
        this->mRequestTimeoutMs = cfg.requestTimeoutMs > 0 ? cfg.requestTimeoutMs : 5000;
        return this->startWorkerLocked(cfg);
    }

    RadarThreeRecognizeResult RadarThreeRecognizerBridge::recognize(const QByteArray& dclsEchoBlob)
    {
        QMutexLocker locker(&this->mMutex);
        RadarThreeRecognizeResult out;
        if(!this->mReady)
        {
            out.errorCode = "NOT_READY";
            out.errorMsg = this->mLastError.isEmpty() ? "bridge is not initialized" : this->mLastError;
            return out;
        }
        if(dclsEchoBlob.isEmpty())
        {
            out.errorCode = "EMPTY_INPUT";
            out.errorMsg = "dclsEchoBlob is empty";
            return out;
        }

        QJsonObject payload;
        payload.insert("echo_blob_b64", QString::fromLatin1(dclsEchoBlob.toBase64()));

        QJsonObject data;
        if(!this->sendCommandLocked("recognize_echo", payload, &data))
        {
            out.errorCode = "RECOGNIZE_FAILED";
            out.errorMsg = this->mLastError;
            return out;
        }

        out.ok = true;
        out.predTargetTypeDec = jsonToInt64(data, "pred_target_type", -1);
        out.predTargetTypeHex = jsonToString(data, "pred_target_type_hex", categoryToHex(out.predTargetTypeDec));
        out.predClassIdDec = jsonToInt64(data, "pred_class_id", -1);
        out.predClassIdHex = jsonToString(data, "pred_class_id_hex", categoryToHex(out.predClassIdDec));
        out.predLabel = jsonToString(data, "pred_label", "");
        out.score = jsonToDouble(data, "score", 0.0);
        out.top1Score = jsonToDouble(data, "top1_score", out.score);
        out.top2Score = jsonToDouble(data, "top2_score", 0.0);
        out.margin = jsonToDouble(data, "margin", 0.0);
        out.probabilities = data.value("probabilities").toObject();
        out.probabilitiesByClassId = data.value("probabilities_by_class_id").toObject();
        out.rawData = data;
        return out;
    }

    void RadarThreeRecognizerBridge::shutdown()
    {
        QMutexLocker locker(&this->mMutex);
        this->stopWorkerLocked();
    }

    bool RadarThreeRecognizerBridge::isReady() const
    {
        QMutexLocker locker(&this->mMutex);
        return this->mReady;
    }

    QString RadarThreeRecognizerBridge::lastError() const
    {
        QMutexLocker locker(&this->mMutex);
        return this->mLastError;
    }

    bool RadarThreeRecognizerBridge::sendCommandLocked(const QString& cmd, const QJsonObject& payload, QJsonObject* outData)
    {
        if(this->mProcess == nullptr || this->mProcess->state() != QProcess::Running)
        {
            this->mReady = false;
            this->mLastError = "worker process is not running";
            return false;
        }

        const int reqId = ++this->mRequestSeq;
        QJsonObject req;
        req.insert("id", reqId);
        req.insert("cmd", cmd);
        for(auto it = payload.constBegin(); it != payload.constEnd(); ++it)
        {
            req.insert(it.key(), it.value());
        }

        const QByteArray line = QJsonDocument(req).toJson(QJsonDocument::Compact) + "\n";
        const qint64 written = this->mProcess->write(line);
        if(written != line.size())
        {
            this->mLastError = "failed to write full request to worker";
            return false;
        }
        if(!this->mProcess->waitForBytesWritten(this->mRequestTimeoutMs))
        {
            this->mLastError = "waitForBytesWritten timeout";
            return false;
        }

        QElapsedTimer timer;
        timer.start();
        while(timer.elapsed() < this->mRequestTimeoutMs)
        {
            while(this->mProcess->canReadLine())
            {
                const QByteArray trimmed = this->mProcess->readLine().trimmed();
                if(trimmed.isEmpty())
                {
                    continue;
                }

                QJsonParseError parseErr;
                const QJsonDocument doc = QJsonDocument::fromJson(trimmed, &parseErr);
                if(parseErr.error != QJsonParseError::NoError || !doc.isObject())
                {
                    continue;
                }

                const QJsonObject resp = doc.object();
                const int respId = static_cast<int>(jsonToInt64(resp, "id", -1));
                if(respId != reqId)
                {
                    continue;
                }

                const bool ok = resp.value("ok").toBool(false);
                if(!ok)
                {
                    const QString errCode = jsonToString(resp, "error_code", "WORKER_ERROR");
                    const QString errMsg = jsonToString(resp, "error_msg", "unknown worker error");
                    this->mLastError = QString("[%1] %2").arg(errCode, errMsg);
                    return false;
                }

                if(outData != nullptr)
                {
                    *outData = resp.value("data").toObject();
                }
                return true;
            }

            const qint64 remain = this->mRequestTimeoutMs - timer.elapsed();
            if(remain <= 0)
            {
                break;
            }
            if(!this->mProcess->waitForReadyRead(static_cast<int>(remain)))
            {
                if(this->mProcess->state() != QProcess::Running)
                {
                    const QString stdErr = QString::fromUtf8(this->mProcess->readAllStandardError());
                    this->mLastError = QString("worker exited unexpectedly. stderr=%1").arg(stdErr);
                    this->mReady = false;
                    return false;
                }
            }
        }

        const QString stdErr = QString::fromUtf8(this->mProcess->readAllStandardError());
        this->mLastError = QString("request timeout (%1 ms). stderr=%2")
                           .arg(this->mRequestTimeoutMs)
                           .arg(stdErr);
        return false;
    }

    bool RadarThreeRecognizerBridge::startWorkerLocked(const RadarThreeRecognizerInitCfg& cfg)
    {
        this->stopWorkerLocked();

        const QString workerEntryPath = this->resolveWorkerEntryPath(cfg);
        if(workerEntryPath.isEmpty())
        {
            this->mLastError = "radar three recognizer worker entry not found";
            return false;
        }

        const QFileInfo ckptInfo(cfg.checkpointPath);
        if(!ckptInfo.exists() || !ckptInfo.isFile())
        {
            this->mLastError = QString("checkpoint not found: %1").arg(cfg.checkpointPath);
            return false;
        }
        const QFileInfo workerInfo(workerEntryPath);
        const QString workerDirPath = workerInfo.absolutePath();

        QProcess* proc = new QProcess();
        proc->setProcessChannelMode(QProcess::SeparateChannels);
        proc->setWorkingDirectory(workerDirPath);
        QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
        env.insert("PYTHONIOENCODING", "utf-8");
        env.insert("PYTHONUTF8", "1");
        QStringList pythonPathEntries;
        pythonPathEntries << workerDirPath;
        const QString currentPythonPath = env.value("PYTHONPATH").trimmed();
        if(!currentPythonPath.isEmpty())
        {
            pythonPathEntries << currentPythonPath;
        }
        env.insert("PYTHONPATH", pythonPathEntries.join(QDir::listSeparator()));
        proc->setProcessEnvironment(env);

        QString program;
        QStringList args;
        if(isWorkerExecutablePath(workerEntryPath))
        {
            program = workerEntryPath;
        }
        else
        {
            program = cfg.pythonExecutable.trimmed().isEmpty() ? QString("python") : cfg.pythonExecutable.trimmed();
            args << workerEntryPath;
        }

        proc->start(program, args);
        if(!proc->waitForStarted(cfg.startupTimeoutMs > 0 ? cfg.startupTimeoutMs : 10000))
        {
            this->mLastError = args.isEmpty()
                               ? QString("failed to start worker executable: %1").arg(program)
                               : QString("failed to start worker: %1 %2").arg(program, args.join(" "));
            delete proc;
            return false;
        }

        this->mProcess = proc;
        this->mReady = false;

        QJsonObject initPayload;
        initPayload.insert("checkpoint_path", ckptInfo.absoluteFilePath());
        initPayload.insert("device", cfg.device);
        initPayload.insert("strict_model_load", cfg.strictModelLoad);

        QJsonObject data;
        if(!this->sendCommandLocked("init", initPayload, &data))
        {
            const QString err = this->mLastError;
            this->stopWorkerLocked();
            this->mLastError = err;
            return false;
        }

        this->mReady = true;
        this->mLastError.clear();
        return true;
    }

    void RadarThreeRecognizerBridge::stopWorkerLocked()
    {
        if(this->mProcess != nullptr)
        {
            if(this->mProcess->state() == QProcess::Running)
            {
                QJsonObject dummy;
                this->sendCommandLocked("shutdown", QJsonObject(), &dummy);
                this->mProcess->closeWriteChannel();
                this->mProcess->terminate();
                if(!this->mProcess->waitForFinished(1000))
                {
                    this->mProcess->kill();
                    this->mProcess->waitForFinished(1000);
                }
            }
            delete this->mProcess;
            this->mProcess = nullptr;
        }
        this->mReady = false;
        this->mRequestSeq = 0;
    }

    QString RadarThreeRecognizerBridge::resolveWorkerEntryPath(const RadarThreeRecognizerInitCfg& cfg) const
    {
        if(!cfg.workerScriptPath.trimmed().isEmpty())
        {
            const QFileInfo fi(cfg.workerScriptPath.trimmed());
            if(fi.exists() && fi.isFile())
            {
                return fi.absoluteFilePath();
            }
        }

        const QStringList relCandidates = {
            "python/radar_three_cls/radar_three_recognizer_worker.py",
            "python/radar_three_cls/radar_three_recognizer_worker.exe",
            "python/radar_three_cls/radar_three_recognizer_worker",
            "radar_three_cls/radar_three_recognizer_worker.py",
            "radar_three_cls/radar_three_recognizer_worker.exe",
            "radar_three_cls/radar_three_recognizer_worker",
            "radar_three_recognizer_worker.py",
            "radar_three_recognizer_worker.exe",
            "radar_three_recognizer_worker",
            "recognizer/radar_three_recognizer_worker.py",
            "recognizer/radar_three_recognizer_worker.exe",
            "recognizer/radar_three_recognizer_worker",
        };

        QStringList candidates;
        for(const QString& rel : relCandidates)
        {
            candidates << QDir::current().absoluteFilePath(rel);
        }

        QDir appDir(QCoreApplication::applicationDirPath());
        for(const QString& rel : relCandidates)
        {
            candidates << appDir.absoluteFilePath(rel);
        }

        QDir walkDir = appDir;
        for(int i = 0; i < 8; ++i)
        {
            for(const QString& rel : relCandidates)
            {
                candidates << walkDir.absoluteFilePath(rel);
            }
            if(!walkDir.cdUp())
            {
                break;
            }
        }

        for(const QString& path : candidates)
        {
            const QFileInfo fi(path);
            if(fi.exists() && fi.isFile())
            {
                return fi.absoluteFilePath();
            }
        }
        return "";
    }

    bool RadarThreeRecognizerBridge::isWorkerExecutablePath(const QString& path)
    {
        const QFileInfo fi(path);
        if(!fi.exists() || !fi.isFile())
        {
            return false;
        }
        if(fi.suffix().compare("exe", Qt::CaseInsensitive) == 0)
        {
            return true;
        }
        if(fi.suffix().compare("py", Qt::CaseInsensitive) == 0)
        {
            return false;
        }

#ifdef Q_OS_WIN
        return false;
#else
        return fi.isExecutable();
#endif
    }

    QString RadarThreeRecognizerBridge::categoryToHex(qint64 value)
    {
        if(value < 0)
        {
            return QString::number(value);
        }
        return QString("0x") + QString::number(static_cast<qulonglong>(value), 16).toUpper();
    }
}
