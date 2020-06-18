package org.apache.ignite.ml.inference.exchange;

import java.nio.file.Path;

public interface MLWritable {
    void save(Path path, ModelFormat mdlFormat);
}
