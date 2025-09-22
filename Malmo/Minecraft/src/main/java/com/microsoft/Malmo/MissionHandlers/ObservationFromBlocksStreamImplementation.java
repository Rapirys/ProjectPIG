
package com.microsoft.Malmo.MissionHandlers;

import com.microsoft.Malmo.MissionHandlerInterfaces.IBinaryDataProducer;
import com.microsoft.Malmo.Schemas.MissionInit;
import io.netty.buffer.Unpooled;
import it.unimi.dsi.fastutil.longs.Long2ObjectMap;
import net.minecraft.block.Block;
import net.minecraft.block.state.IBlockState;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ChunkProviderClient;
import net.minecraft.client.multiplayer.WorldClient;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.network.PacketBuffer;
import net.minecraft.util.SoundCategory;
import net.minecraft.util.SoundEvent;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.IWorldEventListener;
import net.minecraft.world.World;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.chunk.storage.ExtendedBlockStorage;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.world.ChunkEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import javax.annotation.Nullable;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Streams Minecraft world updates as:
 *  - Chunk loads (with present sections, palette-packed like vanilla)
 *  - Chunk unloads
 *  - Sparse single-block updates (x, y, z, stateId)
 *
 * Section payload (per present section):
 *   u8   sectionY
 *   u32  sectionBytes
 *   <sectionBytes> bytes written by ExtendedBlockStorage.getData().write(PacketBuffer)
 *
 * Message header (this blob only; outer framing is done by caller):
 *   u32  MAGIC
 *   u32  n_chunk_loads
 *   u32  n_chunk_unloads
 *   u32  n_block_updates
 *
 * For each chunk load:
 *   i32  chunk_x
 *   i32  chunk_z
 *   u16  section_mask (bit y set -> section y included)
 *   [sections...]
 *
 * For each chunk unload:
 *   i32  chunk_x
 *   i32  chunk_z
 *
 * For each single-block update:
 *   i32  x
 *   i32  y
 *   i32  z
 *   u16  state_id
 */
public class ObservationFromBlocksStreamImplementation extends HandlerBase implements IBinaryDataProducer {

    private static final Log LOG = LogFactory.getLog(ObservationFromBlocksStreamImplementation.class);

    private static final int MAGIC = 0x4B534C42; // 'B''L''S''K'

    /** Packed chunk keys (cx,cz) that were loaded this tick. */
    private final Set<Long> chunkLoads = new HashSet<>();

    /** Packed chunk keys (cx,cz) that were unloaded this tick. */
    private final Set<Long>  chunkUnloads = new HashSet<>();

    /** Sparse single-block updates collected this tick. */
    private final List<BlockUpdate> blockUpdates = new ArrayList<>();

    private final ClientBlockChangeListener blockChangeListener = new ClientBlockChangeListener();
    private final ClientChunkListener chunkListener = new ClientChunkListener();

    private Field loadedChunksField;

    @Override
    public void prepare(final MissionInit missionInit) {
        final WorldClient world = Minecraft.getMinecraft().world;
        if (world == null) return;

        world.addEventListener(blockChangeListener);
        MinecraftForge.EVENT_BUS.register(chunkListener);

        clearTickBuffers();
        LOG.debug("[BlocksStream] Prepared and listeners registered.");
    }

    @Override
    public void cleanup() {
        final WorldClient world = Minecraft.getMinecraft().world;
        if (world != null) {
            world.removeEventListener(blockChangeListener);
        }
        MinecraftForge.EVENT_BUS.unregister(chunkListener);

        clearTickBuffers();
        LOG.debug("[BlocksStream] Cleaned up and listeners unregistered.");
    }

    @Override
    public void writeBinaryData(final MissionInit missionInit, final ByteBuffer out) {
        final WorldClient world = Minecraft.getMinecraft().world;
        if (world == null) return;

        if (CommandsForRequestingObservationImplementation.ObservationRequestBus.consumeWorld()) {
            markLoadedSections(world);
        }

        out.order(ByteOrder.LITTLE_ENDIAN);

        final int startPos = out.position();

        // --- Header (MAGIC + count placeholders we fill later) ---
        out.putInt(MAGIC);
        out.putInt(chunkLoads.size()); // n_chunk_loads
        out.putInt(chunkUnloads.size()); // n_chunk_unloads
        out.putInt(blockUpdates.size()); // n_block_updates

        // --- Chunk loads ---
        for (long key : chunkLoads) {
            final int cx = unpackX(key);
            final int cz = unpackZ(key);
            out.putInt(cx);
            out.putInt(cz);

            // Placeholder for section mask; we patch after we know which sections we emitted.
            final int maskPos = out.position();
            out.putShort((short) 0);
            short sectionMask = 0;

            final Chunk chunk = world.getChunkProvider().getLoadedChunk(cx, cz);
            if (chunk == null) {
                throw new IllegalStateException("Chunk is null on load event: (" + cx + "," + cz + ")");
            }
            final ExtendedBlockStorage[] stores = chunk.getBlockStorageArray();
            for (int sy = 0; sy < 16; sy++) {
                final ExtendedBlockStorage ebs = (sy >= 0 && sy < stores.length) ? stores[sy] : null;
                if (ebs == null || ebs.isEmpty()) continue;

                sectionMask |= (1 << sy);

                out.put((byte) sy);

                // Reserve space for section size and then write palette-packed bytes.
                final int sectionLenngtPosition = out.position();
                out.putInt(0); // section_bytes placeholder
                final int dataBeginningPosition = out.position();
                writeSectionPalettePacked(ebs, out);

                out.putInt(sectionLenngtPosition, out.position() - dataBeginningPosition);
            }


            out.putShort(maskPos, sectionMask);
        }

        // ----- Chunk unloads -----
        for (long key : chunkUnloads) {
            out.putInt(unpackX(key));
            out.putInt(unpackZ(key));
        }

        // --- Single-block updates ---
        for (BlockUpdate bu : blockUpdates) {
            out.putInt(bu.x);
            out.putInt(bu.y);
            out.putInt(bu.z);
            out.putShort((short) (bu.stateId & 0xFFFF));
        }

        LOG.debug(String.format(
                "[BlocksStream] wrote update: chunk loads=%d chunk unloads=%d single blocks updates=%d bytes=%d",
                chunkLoads.size(), chunkUnloads.size(), blockUpdates.size(), (out.position() - startPos)
        ));

        clearTickBuffers();
    }

    // === Listeners ==========================================================

    private final class ClientBlockChangeListener implements IWorldEventListener {
        @Override
        public void notifyBlockUpdate(World worldIn, BlockPos pos, IBlockState oldState, IBlockState newState, int flags) {
            final int stateId = Block.getStateId(newState);
            blockUpdates.add(new BlockUpdate(pos.getX(), pos.getY(), pos.getZ(), stateId));
        }

        @Override public void notifyLightSet(BlockPos pos) {}
        @Override public void markBlockRangeForRenderUpdate(int x1, int y1, int z1, int x2, int y2, int z2) {}
        @Override public void playSoundToAllNearExcept(@Nullable EntityPlayer player, SoundEvent soundIn, SoundCategory category, double x, double y, double z, float volume, float pitch) {}
        @Override public void playRecord(SoundEvent soundIn, BlockPos pos) {}
        @Override public void spawnParticle(int particleID, boolean ignoreRange, double xCoord, double yCoord, double zCoord, double xSpeed, double ySpeed, double zSpeed, int... parameters) {}
        @Override public void spawnParticle(int id, boolean ignore1, boolean ignore2, double x, double y, double z, double vx, double vy, double vz, int... params) {}
        @Override public void onEntityAdded(Entity entityIn) {}
        @Override public void onEntityRemoved(Entity entityIn) {}
        @Override public void broadcastSound(int soundID, BlockPos pos, int data) {}
        @Override public void playEvent(EntityPlayer player, int type, BlockPos blockPosIn, int data) {}
        @Override public void sendBlockBreakProgress(int breakerId, BlockPos pos, int progress) {}
    }

    private final class ClientChunkListener {
        @SubscribeEvent
        public void onLoad(final ChunkEvent.Load event) {
            if (!event.getWorld().isRemote) return;
            final Chunk c = event.getChunk();
            chunkLoads.add(pack(c.xPosition, c.zPosition));
        }

        @SubscribeEvent
        public void onUnload(final ChunkEvent.Unload event) {
            if (!event.getWorld().isRemote) return;
            final Chunk c = event.getChunk();
            chunkUnloads.add(pack(c.xPosition, c.zPosition));
        }
    }

    // === Helpers ============================================================
    private static long pack(int cx, int cz) { return (((long) cx) << 32) ^ (cz & 0xFFFFFFFFL); }
    private static int unpackX(long key) { return (int) (key >> 32); }
    private static int unpackZ(long key) { return (int) key; }

    private void clearTickBuffers() {
        chunkLoads.clear();
        chunkUnloads.clear();
        blockUpdates.clear();
    }

    /**
     * Writes palette-packed section bytes into {@code out}.
     * Uses vanilla DataPaletteBlock serialization (palette + bit array).
     */
    private static void writeSectionPalettePacked(final ExtendedBlockStorage ebs, final ByteBuffer out) {
        final PacketBuffer buf = new PacketBuffer(Unpooled.buffer(8192));
        try {
            ebs.getData().write(buf);
            final int n = buf.readableBytes();
            final byte[] tmp = new byte[n];
            buf.readBytes(tmp);
            out.put(tmp);
        } finally {
            buf.release();
        }
    }

    /** POD for sparse single-block updates. */
    private static final class BlockUpdate {
        final int x, y, z;
        final int stateId;
        BlockUpdate(final int x, final int y, final int z, final int stateId) {
            this.x = x; this.y = y; this.z = z; this.stateId = stateId;
        }
    }

    private void markLoadedSections(WorldClient world) {
        for (Chunk chunk : loadedChunks(world)) {
            if (chunk != null) {
                chunkLoads.add(pack(chunk.xPosition, chunk.zPosition));
            }
        }
    }


    //TODO Move to utils
     private Iterable<Chunk> loadedChunks(WorldClient world) {
        try {
            if (loadedChunksField == null) {
                this.loadedChunksField = ChunkProviderClient.class.getDeclaredField("chunkMapping");
                loadedChunksField.setAccessible(true);
            }
            Long2ObjectMap<Chunk> map = (Long2ObjectMap<Chunk>) loadedChunksField.get(world.getChunkProvider());
            return map.values();
        } catch (Exception e) {
            LOG.error("Failed to get loadedChunks", e);
            throw new RuntimeException(e);
        }
    }
}
