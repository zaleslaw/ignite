/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.ignite.ml.math.exceptions.preprocessing;

import org.apache.ignite.IgniteException;

/**
 * Indicates an illegal feature type and value.
 */
public class IllegalFeatureTypeException extends IgniteException {
    /** */
    private static final long serialVersionUID = 12342524542660L;

    /**
     * @param illegalClass The illegal type.
     * @param illegalVal The illegal value.
     * @param desiredClass The desired type.
     */
    public IllegalFeatureTypeException(Class illegalClass, Object illegalVal, Class desiredClass) {
        super("The type of feature " + illegalClass + " is illegal. The found value is: " + illegalVal + " The type of label should be " + desiredClass);
    }
}
