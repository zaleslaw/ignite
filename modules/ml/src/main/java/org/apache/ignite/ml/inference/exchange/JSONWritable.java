package org.apache.ignite.ml.inference.exchange;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

public interface JSONWritable {
    default void toJSON(Path path) {
        ObjectMapper mapper = new ObjectMapper();

        try {
            File file = new File(path.toAbsolutePath().toString());
            mapper.writeValue(file, this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
