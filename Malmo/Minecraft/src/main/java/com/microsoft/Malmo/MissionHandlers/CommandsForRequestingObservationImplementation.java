// --------------------------------------------------------------------------------------------------
// Requests coming from Python: "request_world_update" / "request_block_state_registry"
// --------------------------------------------------------------------------------------------------

package com.microsoft.Malmo.MissionHandlers;

import com.microsoft.Malmo.MissionHandlerInterfaces.ICommandHandler;
import com.microsoft.Malmo.Schemas.CommandsForRequestingObservation;
import com.microsoft.Malmo.Schemas.MissionInit;
import com.microsoft.Malmo.Schemas.RequestObservationCommand;

import java.util.concurrent.atomic.AtomicBoolean;

/** Command verbs to explicitly request out-of-band observations. */
public class CommandsForRequestingObservationImplementation extends CommandBase implements ICommandHandler {

    private boolean isOverriding = false;

    @Override
    protected boolean onExecute(String verb, String parameter, MissionInit missionInit) {
        if (verb.equalsIgnoreCase(RequestObservationCommand.REQUEST_LOADED_CHUNKS.value())) {
            ObservationRequestBus.requestWorld();
            return true;
        }
        if (verb.equalsIgnoreCase(RequestObservationCommand.REQUEST_BLOCK_STATE_REGISTRY.value())) {
            ObservationRequestBus.requestRegistry();
            return true;
        }
        return false;
    }

    @Override
    public boolean parseParameters(Object params) {
        if (params == null || !(params instanceof CommandsForRequestingObservation))
            return false;

        CommandsForRequestingObservation cparams = (CommandsForRequestingObservation)params;
        setUpAllowAndDenyLists(cparams.getModifierList());
        return true;
    }

    @Override public void install(MissionInit missionInit) {}
    @Override public void deinstall(MissionInit missionInit) {}

    @Override public boolean isOverriding() { return isOverriding; }
    @Override public void setOverriding(boolean b) { isOverriding = b; }



    /** Single-process flags to request one-shot emission on the next tick/peek/step. */
    public static final class ObservationRequestBus {
        private static final AtomicBoolean requestLoadedChunks = new AtomicBoolean(false);
        private static final AtomicBoolean requestBlockStateRegistry = new AtomicBoolean(false);

        public static void requestWorld()   { requestLoadedChunks.set(true); }
        public static void requestRegistry(){ requestBlockStateRegistry.set(true); }

        /** Consume-and-clear: used by the client tick code. */
        public static boolean consumeWorld()    { return requestLoadedChunks.getAndSet(false); }
        public static boolean consumeRegistry() {
            return requestBlockStateRegistry.getAndSet(false);
        }
    }

}
