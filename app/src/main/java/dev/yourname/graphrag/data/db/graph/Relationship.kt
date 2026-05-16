package dev.yourname.graphrag.data.db.graph

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "relationships",
    indices = [
        Index("fromId"),
        Index("toId")
    ]
)
data class Relationship(
    @PrimaryKey val id: String,
    val fromId: String,
    val toId: String,
    val label: String,
    val weight: Float = 1.0f
)

