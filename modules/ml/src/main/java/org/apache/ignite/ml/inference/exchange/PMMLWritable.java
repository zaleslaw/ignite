package org.apache.ignite.ml.inference.exchange;

import java.nio.file.Path;

public interface PMMLWritable {
    void toPMML(Path path);
}
