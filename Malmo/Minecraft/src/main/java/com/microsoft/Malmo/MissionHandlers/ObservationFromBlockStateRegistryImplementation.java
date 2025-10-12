package com.microsoft.Malmo.MissionHandlers;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.microsoft.Malmo.MissionHandlerInterfaces.IObservationProducer;
import com.microsoft.Malmo.Schemas.MissionInit;
import net.minecraft.block.Block;
import net.minecraft.block.properties.IProperty;
import net.minecraft.block.state.IBlockState;
import net.minecraft.util.ResourceLocation;

import java.util.*;

/**
 * Emits a one-shot JSON registry that maps global block-state IDs (1.8–1.12)
 * back to (block_id, meta) and provides readable names.
 *
 * JSON shape:
 * {
 *   "BlockStateRegistry": {
 *     "protocol": "1.12",
 *     "global_state_count": <int>,         // total size of BLOCK_STATE_IDS
 *     "block_count": <int>,                // distinct block_ids encountered
 *     "blocks": [
 *       {
 *         "block_id": <int>,
 *         "name": "minecraft:stone",
 *         "meta_count": <int>,             // number of meta values present
 *         "metas": [
 *           { "meta": 0, "global_id": 1, "state": "minecraft:stone[variant=stone]" },
 *           { "meta": 1, "global_id": 3, "state": "minecraft:stone[variant=granite]" },
 *           ...
 *         ]
 *       },
 *       ...
 *     ]
 *   }
 * }
 */
public class ObservationFromBlockStateRegistryImplementation extends HandlerBase implements IObservationProducer {
    private JsonObject cached;

    @Override
    public void prepare(MissionInit missionInit) {
        cached = buildRegistryJson();
    }

    @Override
    public void cleanup() {
        cached = null;
    }

    @Override
    public void writeObservationsToJSON(JsonObject out, MissionInit missionInit) {
        if (cached == null || !CommandsForRequestingObservationImplementation.ObservationRequestBus.consumeRegistry()) return;
        out.add("BlockStateRegistry", cached);
    }

    // ------------ internals ------------

    private static JsonObject buildRegistryJson() {
        // blockId -> MetaInfo map, sorted for stable output
        TreeMap<Integer, BlockMetaInfo> perBlock = new TreeMap<>();

        final int total = Block.BLOCK_STATE_IDS.size(); // total number of entries
        for (int gid = 0; gid < total; gid++) {
            IBlockState s = Block.BLOCK_STATE_IDS.getByValue(gid);
            if (s == null) continue; // defensive (shouldn't happen in 1.12)

            Block b = s.getBlock();
            int blockId = Block.getIdFromBlock(b);
            int meta    = b.getMetaFromState(s);

            BlockMetaInfo info = perBlock.computeIfAbsent(blockId, k -> new BlockMetaInfo(b));
            info.addMeta(meta, gid, s);
        }

        JsonArray blocksArr = new JsonArray();
        for (Map.Entry<Integer, BlockMetaInfo> e : perBlock.entrySet()) {
            int blockId = e.getKey();
            BlockMetaInfo info = e.getValue();

            JsonObject bj = new JsonObject();
            bj.addProperty("block_id", blockId);
            bj.addProperty("name", info.registryName());
            bj.addProperty("meta_count", info.metaCount());

            JsonArray metas = new JsonArray();
            for (Map.Entry<Integer, MetaEntry> m : info.metas.entrySet()) {
                JsonObject mj = new JsonObject();
                mj.addProperty("meta", m.getKey());
                mj.addProperty("global_id", m.getValue().globalId);
                mj.addProperty("state", m.getValue().stateString);
                metas.add(mj);
            }
            bj.add("metas", metas);
            blocksArr.add(bj);
        }

        JsonObject root = new JsonObject();
        root.addProperty("protocol", "1.12");
        root.addProperty("global_state_count", total);
        root.addProperty("block_count", perBlock.size());
        root.add("blocks", blocksArr);
        return root;
    }

    // Holds per-block meta information
    private static final class BlockMetaInfo {
        final Block block;
        final TreeMap<Integer, MetaEntry> metas = new TreeMap<>();

        BlockMetaInfo(Block b) {
            this.block = b;
        }

        void addMeta(int meta, int globalId, IBlockState state) {
            metas.put(meta, new MetaEntry(globalId, stateToString(state)));
        }

        String registryName() {
            ResourceLocation rl = block.getRegistryName();
            return rl != null ? rl.toString() : "unknown";
        }

        int metaCount() { return metas.size(); }
    }

    private static final class MetaEntry {
        final int globalId;
        final String stateString;

        MetaEntry(int gid, String s) {
            this.globalId = gid;
            this.stateString = s;
        }
    }

    // Produce strings like: "minecraft:stone[variant=granite,powered=false]"
    private static String stateToString(IBlockState state) {
        ResourceLocation registryKey = state.getBlock().getRegistryName();
        String blockName = (registryKey != null) ? registryKey.toString() : "unknown";

        Map<IProperty<?>, Comparable<?>> properties = state.getProperties();
        if (properties.isEmpty()) {
            return blockName;
        }

        // Sort for determinism: variant before powered, etc.
        List<Map.Entry<IProperty<?>, Comparable<?>>> entries =
                new ArrayList<>(properties.entrySet());
        entries.sort(Comparator.comparing(e -> e.getKey().getName()));

        StringBuilder sb = new StringBuilder(blockName).append('[');
        boolean first = true;
        for (Map.Entry<IProperty<?>, Comparable<?>> entry : entries) {
            if (!first) sb.append(',');
            first = false;

            IProperty<?> property = entry.getKey();
            Comparable<?> value = entry.getValue();

            sb.append(property.getName())
                    .append('=')
                    .append(propertyValueName(property, value));
        }
        sb.append(']');
        return sb.toString();
    }

    /** Bridge away wildcard capture so we can call IProperty#getName(T) safely. */
    @SuppressWarnings({"rawtypes", "unchecked"})
    private static String propertyValueName(IProperty property, Comparable value) {
        return property.getName(value);
    }

}
