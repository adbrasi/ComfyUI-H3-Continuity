# Máscara apenas no áudio fornecido

`missing_audio=generate` é o novo padrão quando soundtrack está conectado. A máscara preserva somente o intervalo coberto pela gravação e pelo contexto sonoro disponível. Trechos sem áudio na janela ficam livres. As bordas usam cobertura proporcional dos tokens de 25 ms. `silence` mantém o comportamento do checkpoint anterior.

O Assemble combina o decode do sampler com as amostras fornecidas nos intervalos exatos. Aproveita também o áudio do contexto para a parte correspondente do vídeo de origem. Áudio fora da janela não é inventado: o relatório informa quanto da origem ficou sem gravação e sem participação na geração. Silêncio dentro de um arquivo é preservado; o Import distingue ausência de faixa de áudio de amostras silenciosas.

76 testes de código passaram. Três renders FL2VA + Turbo, 8 steps, Euler/simple, 736×416, origem de 72 frames (3 s), contexto22, alvo124, entrega174 frames (7,25 s):

- full_video com gravação de 0,5 s: gravação no início; áudio gerado na janela; lacuna anterior à janela permanece silenciosa.
- continuation_only com gravação de 0,5 s e sem source_audio: áudio gerado no contexto, gravação entre 3 e 3,5 s, áudio gerado após isso.
- full_video com gravação de 4 s: gravação em [0,4) s e áudio gerado de 4 até 7,25 s.

Todos os FLACs montados correspondem exatamente à combinação da gravação original e do decode exportado nos intervalos esperados (erro máximo zero). Os trechos livres contêm áudio não nulo. Isso verifica a máscara e a montagem, não qualidade perceptual nem sincronização labial. MP4 pode comprimir som; os testes de amostras usam FLAC.

Para reproduzir, restaure h3_source_3s.mp4 pelo checkpoint research_retake_01 e copie os dois WAVs de artifacts/ para input/. Execute o grafo API desejado. Modelos/hashes são os do Marco01. Resultados e tempos estão em results.json; o primeiro inclui carregamento dos modelos.

Os exemplos normais atualizados estão em workflows/03_audio_full_video.json e 04_audio_continuation_only.json. [Conexões e limites](../../docs/AUDIO_ORIGINAL.md). Snapshots anteriores não foram alterados.
