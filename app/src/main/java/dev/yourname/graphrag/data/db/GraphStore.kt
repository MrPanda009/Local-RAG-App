package dev.yourname.graphrag.data.db

import androidx.sqlite.db.SimpleSQLiteQuery
import androidx.sqlite.db.SupportSQLiteQuery
import dev.yourname.graphrag.data.db.graph.Entity
import dev.yourname.graphrag.data.db.graph.GraphDao
import dev.yourname.graphrag.data.db.graph.Relationship
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class GraphStore(private val graphDao: GraphDao) {

    suspend fun insertEntity(entity: Entity) = withContext(Dispatchers.IO) {
        graphDao.insertEntity(entity)
    }

    suspend fun insertRelationship(rel: Relationship) = withContext(Dispatchers.IO) {
        graphDao.insertRelationship(rel)
    }

    suspend fun getNeighborsWithinDepth(
        startId: String,
        depth: Int = 2
    ): List<String> = withContext(Dispatchers.IO) {
        val query = buildNeighborQuery(startId, depth)
        graphDao.getNeighbors(query)
    }

    private fun buildNeighborQuery(startId: String, depth: Int): SupportSQLiteQuery {
        val sql = """
            WITH RECURSIVE neighbors AS (
                SELECT fromId, toId, 1 AS hop_depth
                FROM relationships
                WHERE fromId = ?
                
                UNION
                
                SELECT r.fromId, r.toId, n.hop_depth + 1
                FROM relationships r
                INNER JOIN neighbors n ON r.fromId = n.toId
                WHERE n.hop_depth < ?
            )
            SELECT DISTINCT toId FROM neighbors
        """.trimIndent()

        return SimpleSQLiteQuery(sql, arrayOf(startId, depth))
    }
}

