package com.example.llama

import android.app.ActivityManager
import android.content.Context
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.WindowManager
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale
import java.util.UUID

/**
 * E4B CPU test: upstream llama.cpp's Android example, changed to
 *  - load the first *.gguf found in the app's external files directory (no picker, no copy),
 *  - show a live line with RAM use and the speed of the last turn,
 *  - accept a prompt from an intent extra, so a run can be started from adb.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var statsTv: TextView
    private lateinit var messagesRv: RecyclerView
    private lateinit var userInputEt: EditText
    private lateinit var userActionFab: FloatingActionButton

    @Volatile private var engine: InferenceEngine? = null
    private var generationJob: Job? = null

    private var isModelReady = false
    private val messages = mutableListOf<Message>()
    private val lastAssistantMsg = StringBuilder()
    private val messageAdapter = MessageAdapter(messages)

    @Volatile private var modelNote = "starting"
    @Volatile private var sendAtMs = 0L
    @Volatile private var firstTextMs = 0L
    @Volatile private var generating = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onBackPressedDispatcher.addCallback { moveTaskToBack(true) }

        statsTv = findViewById(R.id.gguf)
        messagesRv = findViewById(R.id.messages)
        messagesRv.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        messagesRv.adapter = messageAdapter
        userInputEt = findViewById(R.id.user_input)
        userActionFab = findViewById(R.id.fab)

        userInputEt.isEnabled = false
        userInputEt.hint = "Loading the model..."
        userActionFab.isEnabled = false
        userActionFab.setOnClickListener { if (isModelReady) handleUserInput() }

        // Engine start + model load
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val e = AiChat.getInferenceEngine(applicationContext)
                engine = e
                e.state.first {
                    it is InferenceEngine.State.Initialized || it is InferenceEngine.State.ModelReady ||
                        it is InferenceEngine.State.Error
                }
                autoLoad(e)
            } catch (t: Throwable) {
                Log.e(TAG, "start failed", t)
                modelNote = "FAILED: ${t.javaClass.simpleName} ${t.message ?: ""}"
            }
        }

        // The stats line, once a second
        lifecycleScope.launch(Dispatchers.Default) {
            var n = 0
            while (isActive) {
                val line = try { buildStats() } catch (t: Throwable) { "stats: ${t.message}" }
                withContext(Dispatchers.Main) { statsTv.text = line }
                if (generating && n % 10 == 0) Log.i(TAG, line.replace('\n', ' '))
                n++
                delay(1000)
            }
        }
    }

    private suspend fun autoLoad(e: InferenceEngine) {
        if (e.state.value is InferenceEngine.State.ModelReady) {
            modelNote = "model already loaded"
        } else {
            val dir = getExternalFilesDir(null)
            val gguf = dir?.listFiles()?.filter { it.isFile && it.name.endsWith(".gguf", ignoreCase = true) }
                ?.maxByOrNull { it.length() }
            if (gguf == null) {
                modelNote = "No .gguf file in ${dir?.path}. Put the model there and reopen the app."
                withContext(Dispatchers.Main) { userInputEt.hint = "No model file" }
                return
            }
            modelNote = "loading ${gguf.name}"
            val t0 = SystemClock.elapsedRealtime()
            e.loadModel(gguf.path)
            val dt = (SystemClock.elapsedRealtime() - t0) / 1000.0
            modelNote = String.format(Locale.US, "%s (loaded in %.1f s)", gguf.name, dt)
            Log.i(TAG, modelNote)
        }
        withContext(Dispatchers.Main) {
            isModelReady = true
            userInputEt.hint = "Type and send a message!"
            userInputEt.isEnabled = true
            userActionFab.setImageResource(R.drawable.outline_send_24)
            userActionFab.isEnabled = true
        }
        // adb: am start -n <pkg>/com.example.llama.MainActivity --es prompt "..." [--ei predict 128]
        //      or --es prompt_file <a text file inside the app's external files directory>
        val p = intent?.getStringExtra("prompt")
            ?: intent?.getStringExtra("prompt_file")?.let { path ->
                try { File(path).readText() } catch (t: Throwable) { Log.e(TAG, "prompt_file", t); null }
            }
        if (!p.isNullOrBlank()) {
            val predict = intent?.getIntExtra("predict", InferenceEngine.DEFAULT_PREDICT_LENGTH)
                ?: InferenceEngine.DEFAULT_PREDICT_LENGTH
            withContext(Dispatchers.Main) { sendMessage(p, predict) }
        }
    }

    private fun handleUserInput() {
        val userMsg = userInputEt.text.toString()
        if (userMsg.isEmpty()) {
            Toast.makeText(this, "Input message is empty!", Toast.LENGTH_SHORT).show()
        } else {
            userInputEt.text = null
            sendMessage(userMsg, InferenceEngine.DEFAULT_PREDICT_LENGTH)
        }
    }

    private fun sendMessage(userMsg: String, predict: Int) {
        val e = engine ?: return
        userInputEt.isEnabled = false
        userActionFab.isEnabled = false

        val shown = if (userMsg.length > 600) userMsg.take(600) + " ... [${userMsg.length} characters]" else userMsg
        messages.add(Message(UUID.randomUUID().toString(), shown, true))
        lastAssistantMsg.clear()
        messages.add(Message(UUID.randomUUID().toString(), lastAssistantMsg.toString(), false))
        messageAdapter.notifyDataSetChanged()
        messagesRv.scrollToPosition(messages.size - 1)

        sendAtMs = SystemClock.elapsedRealtime()
        firstTextMs = 0L
        generating = true
        generationJob = lifecycleScope.launch(Dispatchers.Default) {
            e.sendUserPrompt(userMsg, predict)
                .onCompletion {
                    generating = false
                    Log.i(TAG, "TURN_DONE " + buildStats().replace('\n', ' '))
                    withContext(Dispatchers.Main) {
                        userInputEt.isEnabled = true
                        userActionFab.isEnabled = true
                    }
                }.collect { token ->
                    if (firstTextMs == 0L) firstTextMs = SystemClock.elapsedRealtime()
                    withContext(Dispatchers.Main) {
                        val messageCount = messages.size
                        check(messageCount > 0 && !messages[messageCount - 1].isUser)
                        messages.removeAt(messageCount - 1).copy(
                            content = lastAssistantMsg.append(token).toString()
                        ).let { messages.add(it) }
                        messageAdapter.notifyItemChanged(messages.size - 1)
                    }
                }
        }
    }

    // ---- the stats line -------------------------------------------------------------------------------------------

    private fun procStatusKb(): Map<String, Long> {
        val out = HashMap<String, Long>()
        try {
            File("/proc/self/status").forEachLine { l ->
                val i = l.indexOf(':')
                if (i > 0) {
                    val k = l.substring(0, i)
                    if (k == "VmRSS" || k == "RssAnon" || k == "RssFile" || k == "RssShmem" || k == "VmSwap") {
                        out[k] = l.substring(i + 1).trim().split(' ')[0].toLongOrNull() ?: 0L
                    }
                }
            }
        } catch (_: Throwable) { }
        return out
    }

    private fun memAvailableMb(): Long {
        try {
            File("/proc/meminfo").useLines { ls ->
                for (l in ls) if (l.startsWith("MemAvailable:")) {
                    return (l.substring(13).trim().split(' ')[0].toLongOrNull() ?: 0L) / 1024
                }
            }
        } catch (_: Throwable) { }
        val mi = ActivityManager.MemoryInfo()
        (getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(mi)
        return mi.availMem / (1024 * 1024)
    }

    private fun buildStats(): String {
        val st = procStatusKb()
        val anon = (st["RssAnon"] ?: 0L) / 1024
        val file = (st["RssFile"] ?: 0L) / 1024
        val swap = (st["VmSwap"] ?: 0L) / 1024
        val free = memAvailableMb()
        val sb = StringBuilder()
        sb.append("E4B CPU test: ").append(modelNote).append('\n')
        sb.append("RAM  own ").append(anon).append(" MB (+").append(swap).append(" swapped) | model pages in cache ")
            .append(file).append(" MB (the system can drop these) | phone available ").append(free).append(" MB\n")
        val parts = (engine?.stats() ?: "").split('|')
        if (parts.size >= 7) {
            val pn = parts[0].toIntOrNull() ?: 0
            val pfS = parts[1].toDoubleOrNull() ?: 0.0
            val gn = parts[2].toIntOrNull() ?: 0
            val genS = parts[3].toDoubleOrNull() ?: 0.0
            val wallS = parts[4].toDoubleOrNull() ?: 0.0
            sb.append("context ").append(parts[5]).append(" / ").append(parts[6]).append(" tokens\n")
            if (pn > 0) {
                sb.append(String.format(Locale.US, "last turn: read %d tok in %.1f s (%.1f tok/s)", pn, pfS,
                    if (pfS > 0) pn / pfS else 0.0))
                if (firstTextMs > 0L && sendAtMs > 0L) {
                    sb.append(String.format(Locale.US, " | first text after %.1f s", (firstTextMs - sendAtMs) / 1000.0))
                }
                if (gn > 0 && genS > 0) {
                    sb.append(String.format(Locale.US, " | wrote %d tok at %.2f tok/s", gn, gn / genS))
                    if (gn > 1 && wallS > 0) sb.append(String.format(Locale.US, " (%.2f with the screen)", (gn - 1) / wallS))
                }
            } else if (generating) {
                sb.append("reading the prompt...")
            }
        }
        return sb.toString()
    }

    override fun onDestroy() {
        if (isFinishing) {
            generationJob?.cancel()
            engine?.destroy()
        }
        super.onDestroy()
    }

    companion object {
        private const val TAG = "E4BCPU"
    }
}
