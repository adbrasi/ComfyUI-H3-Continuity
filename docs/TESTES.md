# Validação da versão 0.1.0

Ambiente: RTX 5090 32 GB; ComfyUI 0.34.0, commit em `revisions.json`. Todos os renders desta validação usam **FL2VA INT8 ConvRot**, **Turbo FL2V 8step v1.0**, **8 steps**, Euler/simple, 736×416. Não houve avaliação perceptual com REF2VA ou sem Turbo. As referências REF2VA são preservadas nos testes de estrutura, mas isso não comprova qualidade de geração com esse checkpoint.

## Comparação controlada

Fonte: mulher caminhando ao lado de uma parede de tijolos, 124 frames. Mesmo prompt e seed 23456 entre métodos. Target 124 frames, contexto 22; montagem 226 frames, 9,417 s.

| Método | Diferença na emenda / mediana local¹ | Correlação do áudio no contexto² |
|---|---:|---:|
| anchors | 1,064 | 0,499 |
| pinned_prefix | 1,015 | 0,505 |
| pinned_av, 22 frames | 1,012 | 0,971 |

¹ Diferença absoluta média de pixels entre frames, dividida pela mediana de uma vizinhança da emenda. Um valor próximo de 1 significa que a emenda tem variação semelhante à local; **não prova** qualidade de movimento, ausência de flicker ou identidade correta.

² Correlação do áudio reconstruído no trecho de contexto repetido, usando o alinhamento conhecido e busca de ±25 ms. Não é uma nota de qualidade do áudio futuro. O som do Turbo pode continuar artificial mesmo com alta correlação. No teste de 22 frames, o melhor lag foi zero. Com 39 frames, a correlação ficou em 0,623 e apareceu deslocamento de aproximadamente −8,34 ms: contexto maior não foi automaticamente melhor.

O maior erro numérico no prefixo de vídeo foi 0,914 para anchors e aproximadamente 4,77×10⁻⁷ para a máscara de vídeo. Isso verifica preservação do latent; a reconstrução VAE não deve ser confundida com pixels originais idênticos.

## Vídeo externo de três segundos

Um MP4 de 72 frames foi comparado usando a mesma seed. Com 56 frames de contexto, pinned_prefix e pinned_av tiveram razões de diferença na emenda de 0,980 e 0,967. Passar os 72 frames diretamente ao Add Guide produziu 2,184, com salto visual perceptível.

Esse baseline representa o uso direto do vídeo completo: Add Guide retém seus primeiros 56 frames. **Não é uma comparação contra um Add Guide manualmente preparado com a cauda correta.** O pack automatiza essa preparação, além das máscaras e da montagem.

## Ação e câmera

Duas seeds (34567 e 45678), três métodos, target de 175 frames. Solicitação: continuar andando, parar, abrir a bolsa, retirar uma chave e fazer um arco de câmera com aproximação. O movimento novo permanece livre depois do prefixo.

O usuário assistiu e aprovou explicitamente `stress_anchors_34567`, `stress_pinned_av_34567` e `stress_anchors_45678`. A inspeção das folhas de contato confirma a progressão da ação e do enquadramento, mas não substitui a avaliação em movimento.

Uma segunda continuação a partir do latent de `stress_pinned_av_34567` solicita aproximação de uma porta, uso da chave e entrada. Resultado: **430 frames / 17,917 s**, com duas emendas. O usuário também assistiu e aprovou esse vídeo. Cópia preservada em `examples/continuacao_duas_emendas.mp4`.

## Gravação original

Teste com uma gravação sintética conhecida, mono, 48 kHz. O áudio entra no VAE na taxa correta e seu latent é fixado por máscara. A montagem retorna as amostras originais.

Comparação do WAV de entrada com o FLAC final: **452.000 amostras**, duração correta para 226 frames a 24 fps e **erro máximo zero**. O caso de VAE produzir um passo a menos foi corrigido com lookahead interno, sem acrescentar tempo ao áudio entregue. Isso verifica preservação de amostras; **não é teste de sincronização labial**.

## Testes de código e desempenho

40 testes passaram: cauda na grade temporal, diferentes durações, áudio alinhado, máscaras AV, referências e last-frame mantidos, entradas não modificadas, importação a 24 fps, montagem sem duplicação, gravação/carregamento sem perda de dtype e contenção de caminhos.

Na configuração de teste, o primeiro render de fonte levou 32,30 s. Continuação simples com anchors levou 27,91 s; exemplos posteriores ficaram na ordem de 20–40 s, conforme comprimento, caminho de codificação e cache. São tempos de execução completos observados nos logs, não benchmarks isolados do node. A máscara não adiciona uma segunda passagem de difusão.

Limitações não verificadas: cadeias muito longas, outras resoluções/GPUs/versões, fala/lip sync exigentes, máscaras combinadas de terceiros e recuperação de um objeto que desapareceu antes da cauda. Os experimentos de memória são posteriores à versão estável e não estão incluídos nas promessas desta versão.
