package org.apache.ignite.ml.inference.exchange;

import org.apache.ignite.ml.IgniteModel;
import org.apache.ignite.ml.math.primitives.vector.Vector;

import java.nio.file.Path;

public interface JSONReadable {
    IgniteModel<Vector, ? extends Number> fromJSON(Path path);
}
