These files are originally from /home/weismanal/notebook/2019-02-21/uno/testing

Note this single change to the uno Python file:

-    unoBmk = benchmark.BenchmarkUno(benchmark.file_path, 'uno_default_model.txt', 'keras',
+    #mymodel_common = candle.Benchmark(file_path,os.getenv("DEFAULT_PARAMS_FILE"),'keras',prog='myprog',desc='My model')
+    unoBmk = benchmark.BenchmarkUno(benchmark.file_path, os.getenv("DEFAULT_PARAMS_FILE"), 'keras',
