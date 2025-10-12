package com.microsoft.Malmo.MissionHandlerInterfaces;

import com.microsoft.Malmo.Schemas.MissionInit;
import java.nio.ByteBuffer;

/** General interface for objects producing binary data each tick (eg world state updates). */
public interface IBinaryDataProducer {
    void prepare(MissionInit missionInit);
    void cleanup();
    void writeBinaryData(MissionInit missionInit, ByteBuffer buffer);
}

