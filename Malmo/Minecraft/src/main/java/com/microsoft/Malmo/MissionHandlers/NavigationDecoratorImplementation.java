// --------------------------------------------------------------------------------------------------
//  Copyright (c) 2016 Microsoft Corporation
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
//  associated documentation files (the "Software"), to deal in the Software without restriction,
//  including without limitation the rights to use, copy, modify, merge, publish, distribute,
//  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all copies or
//  substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// --------------------------------------------------------------------------------------------------
package com.microsoft.Malmo.MissionHandlers;

import com.microsoft.Malmo.MissionHandlerInterfaces.IWorldDecorator;
import com.microsoft.Malmo.Schemas.MissionInit;
import com.microsoft.Malmo.Schemas.NavigationDecorator;
import com.microsoft.Malmo.Utils.MinecraftTypeHelper;
import com.microsoft.Malmo.Utils.SeedHelper;
import net.minecraft.block.state.IBlockState;
import net.minecraft.client.Minecraft;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class NavigationDecoratorImplementation extends HandlerBase implements IWorldDecorator {

    private static final int BOUNDARY_SIZE = 64;     // worldborder size (square)
    private static final int HALF = BOUNDARY_SIZE / 2;

    private NavigationDecorator nparams;

    // Chosen positions
    private double originX, originY, originZ;
    private double targetX, targetY, targetZ;
    private boolean compassTargetSet = false;

    @Override
    public boolean parseParameters(Object params) {
        if (params == null || !(params instanceof NavigationDecorator))
            return false;
        this.nparams = (NavigationDecorator) params;
        return true;
    }

    //EntityPlayerSP['Agent1'/632, l='MpServer', x=-37.50, y=67.00, z=97.50]
    //-17.505962044299253, 82.7790571232187

    @Override
    public void buildOnWorld(MissionInit missionInit, World world) throws DecoratorException {
        // --- Preserve original spawn as origin ---
        BlockPos originalSpawn = world.getSpawnPoint();
        originX = originalSpawn.getX();
        originY = originalSpawn.getY();
        originZ = originalSpawn.getZ();

        // --- Random border center, but keep original spawn inside the 100x100 square ---
        // Spawn must satisfy: |spawn - center| < HALF
        double offX = (SeedHelper.getRandom().nextDouble() * 2 - 1) * (HALF - 1);
        double offZ = (SeedHelper.getRandom().nextDouble() * 2 - 1) * (HALF - 1);
        double centerX = originX + offX;
        double centerZ = originZ + offZ;


        MinecraftServer server = world.getMinecraftServer();
        server.getCommandManager().executeCommand(server,
                "worldborder center " + (centerX + 0.5) + " " + (centerZ + 0.5));
        server.getCommandManager().executeCommand(server,
                "worldborder set " + BOUNDARY_SIZE);

//        WorldBorder border = world.getWorldBorder();
//        border.setSize(BOUNDARY_SIZE);
//        border.setCenter(centerX + 0.5, centerZ + 0.5);
//
//        SPacketWorldBorder init = new SPacketWorldBorder(border, SPacketWorldBorder.Action.INITIALIZE);
//        for (EntityPlayerMP p : world.getPlayers(EntityPlayerMP.class, Predicates.alwaysTrue())) {
//            p.connection.sendPacket(init);
//        }


        // 4) Pick target position in boundary (not equal to player’s boundary cell)
        int minX = (int)Math.floor(centerX) - HALF + 1;
        int maxX = (int)Math.floor(centerX) + HALF - 1;
        int minZ = (int)Math.floor(centerZ) - HALF + 1;
        int maxZ = (int)Math.floor(centerZ) + HALF - 1;

        int tx = minX + SeedHelper.getRandom().nextInt(Math.max(1, maxX - minX + 1));
        int tz = minZ + SeedHelper.getRandom().nextInt(Math.max(1, maxZ - minZ + 1));
        targetX = tx; targetZ = tz; targetY = 0;

        // Block to place (default diamond_block if unspecified)
        String blockName = "diamond_block";
        IBlockState state = MinecraftTypeHelper.ParseBlockType(blockName); //TODO will compas point to target

        // 5) Build 3x3x(world_height) column centered at target
        int maxY = world.getActualHeight(); // full vertical build height
        for (int dx = -1; dx <= 1; dx++) {
            for (int dz = -1; dz <= 1; dz++) {
                for (int y = 0; y < maxY; y++) {
                    world.setBlockState(new BlockPos((int)targetX + dx, y, (int)targetZ + dz), state, 2);
                }
            }
        }
    }

    @Override
    public boolean getExtraAgentHandlersAndData(List<Object> handlers, Map<String, String> data) {
        return false;
    }

    @Override
    public void update(World world) {
        if (!compassTargetSet && Minecraft.getMinecraft().player != null) {
            world.setSpawnPoint(new BlockPos((int)targetX, 0, (int)targetZ));
            compassTargetSet = true;
        }
    }
    @Override public void prepare(MissionInit missionInit) {
    }
    @Override public void cleanup() { }
    @Override public boolean targetedUpdate(String nextAgentName) { return false; }
    @Override public void getTurnParticipants(ArrayList<String> participants, ArrayList<Integer> participantSlots) { }
}
