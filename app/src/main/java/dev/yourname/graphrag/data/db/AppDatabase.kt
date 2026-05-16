package dev.yourname.graphrag.data.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.sqlite.db.SupportSQLiteDatabase
import androidx.sqlite.driver.bundled.BundledSQLiteDriver
import dev.yourname.graphrag.data.db.graph.Entity
import dev.yourname.graphrag.data.db.graph.GraphDao
import dev.yourname.graphrag.data.db.graph.Relationship
import kotlinx.coroutines.Dispatchers

@Database(
    entities = [Entity::class, Relationship::class],
    version = 1
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun graphDao(): GraphDao

    companion object {
        fun build(context: Context): AppDatabase {
            return Room.databaseBuilder(
                context,
                AppDatabase::class.java,
                "graph_rag.db"
            )
                .setDriver(BundledSQLiteDriver())
                .setQueryCoroutineContext(Dispatchers.IO)
                .build()
        }
    }
}

