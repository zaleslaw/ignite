package org.apache.ignite.ml.clustering;

/** Status of a point during the clustering process. */
public enum VectorStatus {
    UNKNOWN,
    /** The point has is considered to be noise. */
    NOISE,
    /** The point is already part of a cluster. */
    PART_OF_CLUSTER
}
