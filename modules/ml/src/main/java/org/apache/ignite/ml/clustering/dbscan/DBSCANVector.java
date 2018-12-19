package org.apache.ignite.ml.clustering.dbscan;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import org.apache.ignite.ml.clustering.VectorStatus;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.structures.LabeledVector;

public class DBSCANVector<V extends Vector, L> extends LabeledVector<V, L> {
    public int clusterId = -1;
    public VectorStatus status = VectorStatus.UNKNOWN;

    public DBSCANVector(V vector, L lb) {
        super(vector, lb);
    }

    public DBSCANVector() {
    }

    /** {@inheritDoc} */
    @Override public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        LabeledVector vector1 = (LabeledVector)o;

        if (vector != null ? !vector.equals(vector1.features()) : vector1.features() != null)
            return false;
        return lb != null ? lb.equals(vector1.label()) : vector1.label() == null;
    }

    /** {@inheritDoc} */
    @Override public int hashCode() {
        int res = vector != null ? vector.hashCode() : 0;
        res = 31 * res + (lb != null ? lb.hashCode() : 0);
        return res;
    }

    /** {@inheritDoc} */
    @Override public void writeExternal(ObjectOutput out) throws IOException {
        out.writeObject(vector);
        out.writeObject(lb);
        out.writeInt(clusterId);
        out.writeObject(status);
    }

    /** {@inheritDoc} */
    @Override public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        vector = (V)in.readObject();
        lb = (L)in.readObject();
        clusterId = in.readInt();
        status = (VectorStatus)in.readObject();
    }
}
