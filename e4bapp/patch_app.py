#!/usr/bin/env python3
"""patch_app.py <llama.cpp checkout>

Turns upstream's examples/llama.android (tag b12317) into the "E4B CPU test" app:
  - its own package (com.tarmaska11.e4bcpu), arm64 only;
  - 32k context, 3 threads, micro-batch 128, weights read straight from the mapped file (no repacking);
  - Gemma 4 turns formatted by hand; an end-of-turn token is never decoded, so every later turn opens with "<turn|>\\n";
  - timing counters of the last turn exposed to Kotlin (stats());
  - MainActivity replaced by e4bapp/MainActivity.kt (auto-load from the external files directory, a RAM + speed line).
Every edit asserts that its anchor matched exactly once, so a changed upstream fails the build instead of silently
producing a different app.
"""
import re
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
here = Path(__file__).resolve().parent
app = src / "examples" / "llama.android"


def edit(path, pattern, repl, flags=0, count=1):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    new, n = re.subn(pattern, lambda m: repl, s, flags=flags)
    assert n == count, f"{p}: pattern {pattern!r} matched {n} times, expected {count}"
    p.write_text(new, encoding="utf-8")
    print(f"ok   {p.relative_to(src)}  {pattern[:60]!r}")


# ---- the engine must have the switch that --no-repack sets ------------------------------------------------------------
assert "use_extra_bufts" in (src / "include" / "llama.h").read_text(encoding="utf-8"), "llama.h has no use_extra_bufts"

# ---- Gradle ----------------------------------------------------------------------------------------------------------
edit(app / "app" / "build.gradle.kts", r'applicationId = "[^"]+"', 'applicationId = "com.tarmaska11.e4bcpu"')
edit(app / "lib" / "build.gradle.kts", r'abiFilters \+= listOf\("arm64-v8a", "x86_64"\)', 'abiFilters += listOf("arm64-v8a")')

strings = app / "app" / "src" / "main" / "res" / "values" / "strings.xml"
edit(strings, r'<string name="app_name">[^<]*</string>', '<string name="app_name">E4B CPU test</string>')

# the Activity keeps its state on rotation and similar changes (the example's engine singleton does not survive a recreate)
edit(app / "app" / "src" / "main" / "AndroidManifest.xml", r'android:name="\.MainActivity"',
     'android:name=".MainActivity"\n            android:configChanges="orientation|screenSize|screenLayout|smallestScreenSize|keyboard|keyboardHidden|navigation|uiMode|density|fontScale|locale|layoutDirection"\n            android:windowSoftInputMode="adjustResize"')

# ---- CMake: the configuration measured on the A54 from the command line (no OpenMP, no KleidiAI) ----------------------
cm = app / "lib" / "src" / "main" / "cpp" / "CMakeLists.txt"
edit(cm, r'set\(GGML_CPU_KLEIDIAI ON\)', 'set(GGML_CPU_KLEIDIAI OFF)')
edit(cm, r'set\(GGML_OPENMP ON\)', 'set(GGML_OPENMP OFF)')

# ---- ai_chat.cpp -----------------------------------------------------------------------------------------------------
cpp = app / "lib" / "src" / "main" / "cpp" / "ai_chat.cpp"

edit(cpp, r'#include <unistd\.h>', '#include <unistd.h>\n#include <atomic>\n#include <cstdio>')
edit(cpp, r'DEFAULT_CONTEXT_SIZE(\s*)=\s*8192;', 'DEFAULT_CONTEXT_SIZE    = 32768;')
edit(cpp, r'static common_sampler\s*\* g_sampler;',
     'static common_sampler                   * g_sampler;\n\n'
     '// E4B test app: counters of the last turn (written by the engine thread, read by the UI thread)\n'
     'static std::atomic<long long> g_st_prompt_us{0}, g_st_gen_us{0}, g_st_first_us{0}, g_st_last_us{0};\n'
     'static std::atomic<int>       g_st_prompt_n{0}, g_st_gen_n{0}, g_st_pos{0};')

# weights stay in the mapped file (= --no-repack); otherwise Q4_K is copied into ~2.6 GB of anonymous memory
edit(cpp, r'llama_model_params model_params = llama_model_default_params\(\);',
     'llama_model_params model_params = llama_model_default_params();\n'
     '    model_params.use_extra_bufts = false;')

# 3 threads (measured best on the A54: 4 big cores, one left for the screen)
edit(cpp, r'const int n_threads = std::max\(N_THREADS_MIN,.*?N_THREADS_HEADROOM\)\);', 'const int n_threads = 3;', flags=re.S)
edit(cpp, r'ctx_params\.n_ubatch = BATCH_SIZE;', 'ctx_params.n_ubatch = 128;')

# the chat-template object is not used for formatting any more; do not let its parser take the app down
edit(cpp, r'g_chat_templates = common_chat_templates_init\(g_model, ""\);',
     'try { g_chat_templates = common_chat_templates_init(g_model, ""); }\n'
     '    catch (...) { LOGw("%s: chat template not parsed (unused)", __func__); }')

# user turn: Gemma 4 format by hand
edit(cpp,
     r'std::string formatted_user_prompt\(user_prompt\);.*?'
     r'auto user_tokens = common_tokenize\(g_context, formatted_user_prompt, has_chat_template, has_chat_template\);',
     '// E4B test app: Gemma 4 turns written by hand. A reply never has its end-of-turn token decoded (see\n'
     '    // generateNextToken), so every turn after the first closes the previous reply itself.\n'
     '    const bool first_turn = (current_position == 0);\n'
     '    std::string formatted_user_prompt = std::string(first_turn ? "" : "<turn|>\\n")\n'
     '            + "<|turn>user\\n" + user_prompt + "<turn|>\\n<|turn>model\\n";\n'
     '    env->ReleaseStringUTFChars(juser_prompt, user_prompt);\n\n'
     '    auto user_tokens = common_tokenize(g_context, formatted_user_prompt, first_turn, true);',
     flags=re.S)

edit(cpp,
     r'if \(decode_tokens_in_batches\(g_context, g_batch, user_tokens, current_position, true\)\) \{',
     'const long long t_pf0 = ggml_time_us();\n'
     '    const int pf_rc = decode_tokens_in_batches(g_context, g_batch, user_tokens, current_position, true);\n'
     '    g_st_prompt_us = ggml_time_us() - t_pf0;\n'
     '    g_st_prompt_n = (int) user_tokens.size();\n'
     '    g_st_gen_us = 0; g_st_gen_n = 0; g_st_first_us = 0; g_st_last_us = 0;\n'
     '    g_st_pos = (int) current_position + (int) user_tokens.size();\n'
     '    if (pf_rc) {')

# generation: time each step; stop before decoding an end-of-generation token
edit(cpp,
     r'const auto new_token_id = common_sampler_sample\(g_sampler, g_context, -1\);\s*'
     r'common_sampler_accept\(g_sampler, new_token_id, true\);',
     'const long long t_g0 = ggml_time_us();\n'
     '    const auto new_token_id = common_sampler_sample(g_sampler, g_context, -1);\n'
     '    common_sampler_accept(g_sampler, new_token_id, true);\n\n'
     '    if (llama_vocab_is_eog(llama_model_get_vocab(g_model), new_token_id)) {\n'
     '        LOGd("id: %d,\\tIS EOG!\\nSTOP.", new_token_id);\n'
     '        return nullptr;\n'
     '    }')
edit(cpp,
     r'// Stop if next token is EOG\s*if \(llama_vocab_is_eog\(llama_model_get_vocab\(g_model\), new_token_id\)\) \{.*?'
     r'chat_add_and_format\(ROLE_ASSISTANT, assistant_ss\.str\(\)\);\s*return nullptr;\s*\}',
     '{\n'
     '        const long long t_g1 = ggml_time_us();\n'
     '        g_st_gen_us += t_g1 - t_g0;\n'
     '        g_st_gen_n += 1;\n'
     '        if (g_st_first_us == 0) { g_st_first_us = t_g1; }\n'
     '        g_st_last_us = t_g1;\n'
     '        g_st_pos = (int) current_position;\n'
     '    }',
     flags=re.S)

# the stats call
s = cpp.read_text(encoding="utf-8")
s += r'''

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_statsNative(JNIEnv *env, jobject /*unused*/) {
    // prompt tokens | prefill s | generated tokens | s inside the engine | s first..last token (wall) | position | context
    const int gen_n = g_st_gen_n.load();
    const double wall_s = gen_n > 1 ? (double) (g_st_last_us.load() - g_st_first_us.load()) / 1e6 : 0.0;
    char buf[200];
    snprintf(buf, sizeof(buf), "%d|%.3f|%d|%.3f|%.3f|%d|%d", g_st_prompt_n.load(), (double) g_st_prompt_us.load() / 1e6,
             gen_n, (double) g_st_gen_us.load() / 1e6, wall_s, g_st_pos.load(), DEFAULT_CONTEXT_SIZE);
    return env->NewStringUTF(buf);
}
'''
cpp.write_text(s, encoding="utf-8")
print("ok   ai_chat.cpp  statsNative appended")

# ---- Kotlin: stats() on the engine interface -------------------------------------------------------------------------
kt = app / "lib" / "src" / "main" / "java" / "com" / "arm" / "aichat"
edit(kt / "InferenceEngine.kt", r'    fun destroy\(\)\n',
     '    fun destroy()\n\n    /**\n     * E4B test app: counters of the last turn, separated by "|"\n     */\n    fun stats(): String\n')
impl = kt / "internal" / "InferenceEngineImpl.kt"
edit(impl, r'    @FastNative\n    private external fun shutdown\(\)\n',
     '    @FastNative\n    private external fun shutdown()\n\n    @FastNative\n    private external fun statsNative(): String\n')
edit(impl, r'    override fun destroy\(\) \{',
     '    override fun stats(): String = try { statsNative() } catch (t: Throwable) { "" }\n\n    override fun destroy() {')

# ---- the Activity -----------------------------------------------------------------------------------------------------
dst = app / "app" / "src" / "main" / "java" / "com" / "example" / "llama" / "MainActivity.kt"
assert dst.exists(), dst
shutil.copyfile(here / "MainActivity.kt", dst)
print("ok   MainActivity.kt replaced")

# show what the engine file looks like around the edits (for the build log)
for key in ("use_extra_bufts", "n_threads = 3", "first_turn", "t_g0", "statsNative"):
    assert key in cpp.read_text(encoding="utf-8"), key
print("patch_app: done")
