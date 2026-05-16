package dev.yourname.graphrag.data.db.entities

import io.objectbox.annotation.Entity
import io.objectbox.annotation.Id
import io.objectbox.annotation.HnswIndex
import io.objectbox.annotation.VectorDistanceType

@Entity
data class Chunk(
    @Id var id: Long = 0,
    var text: String = "",
    var sourceUrl: String = "",
    var sourceTitle: String = "",
    var chunkIndex: Int = 0,
    @HnswIndex(dimensions = 384, distanceType = VectorDistanceType.COSINE)
    var embedding: FloatArray? = null
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Chunk

        if (id != other.id) return false
        if (text != other.text) return false
        if (sourceUrl != other.sourceUrl) return false
        if (sourceTitle != other.sourceTitle) return false
        if (chunkIndex != other.chunkIndex) return false
        if (embedding != null) {
            if (other.embedding == null) return false
            if (!embedding.contentEquals(other.embedding)) return false
        } else if (other.embedding != null) return false

        return true
    }

    override fun hashCode(): Int {
        var result = id.hashCode()
        result = 31 * result + text.hashCode()
        result = 31 * result + sourceUrl.hashCode()
        result = 31 * result + sourceTitle.hashCode()
        result = 31 * result + chunkIndex
        result = 31 * result + (embedding?.contentHashCode() ?: 0)
        return result
    }
}

