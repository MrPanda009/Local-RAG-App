package dev.yourname.graphrag.data.db.graph

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.RawQuery
import androidx.sqlite.db.SupportSQLiteQuery

@Dao
interface GraphDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertEntity(entity: Entity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertRelationship(rel: Relationship)

    @Query("SELECT * FROM entities WHERE id = :id")
    suspend fun getEntity(id: String): Entity?

    @Query("SELECT * FROM entities WHERE name LIKE '%' || :query || '%'")
    suspend fun searchEntities(query: String): List<Entity>

    @RawQuery
    suspend fun getNeighbors(query: SupportSQLiteQuery): List<String>
}

