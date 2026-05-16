package dev.yourname.graphrag.data.db.graph

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "entities")
data class Entity(
    @PrimaryKey val id: String,
    val name: String,
    val type: String,
    val description: String,
    val sourceChunkId: Long
)

