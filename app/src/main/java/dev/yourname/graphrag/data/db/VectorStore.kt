package dev.yourname.graphrag.data.db

import dev.yourname.graphrag.data.db.entities.Chunk
import dev.yourname.graphrag.data.db.entities.Chunk_
import io.objectbox.Box
import io.objectbox.BoxStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class VectorStore(boxStore: BoxStore) {
    private val box: Box<Chunk> = boxStore.boxFor(Chunk::class.java)

    suspend fun upsertChunk(chunk: Chunk) = withContext(Dispatchers.IO) {
        box.put(chunk)
    }

    suspend fun upsertChunks(chunks: List<Chunk>) = withContext(Dispatchers.IO) {
        box.put(chunks)
    }

    suspend fun findNearest(
        queryEmbedding: FloatArray,
        limit: Int = 5
    ): List<Chunk> = withContext(Dispatchers.IO) {
        box.query(
            Chunk_.embedding.nearestNeighbors(queryEmbedding, limit)
        ).build().findWithScores().map { it.entity }
    }
}

