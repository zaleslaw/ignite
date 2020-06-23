package org.apache.ignite.ml.inference.exchange;

import org.apache.ignite.ml.IgniteModel;
import org.apache.ignite.ml.math.primitives.vector.Vector;

import java.nio.file.Path;

public interface PMMLReadable {
    IgniteModel<Vector, ? extends Number> fromPMML(Path path);
}
