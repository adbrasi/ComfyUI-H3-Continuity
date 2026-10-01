# Continuação, bridge e retake

O pack usa o sampler H3 já instalado. Prepare organiza os latentes e as máscaras; o sampler gera conteúdo; Assemble escolhe o trecho entregue. Conservar contexto no modelo e conservar frames na saída são decisões distintas: se uma borda foi liberada para reparo, a montagem precisa entregar esse reparo.

## Instalação e formato

Instale pelo [README](../README.md). Use os VAEs de vídeo e áudio H3, os nodes nativos H3 e um ComfyUI com suporte a máscaras AV por token. Nenhum treinamento adicional é necessário.

O vídeo trabalha a **24 fps**, com latentes completos de `5k + 2` passos correspondendo a `17k + 5` frames. O ciclo de suporte temporal é `(1, 4, 4, 4, 4)` frames por passo. Exemplos válidos: 22, 39, 56, 73, 124 e 141 frames. O áudio tem 40 passos por segundo; suas bordas nem sempre coincidem com bordas de vídeo.

Não recorte a saída latente do sampler antes de salvar ou usá-la como origem. Um checkpoint pode conter apenas a última janela gerada, com contexto; não representa necessariamente todo o filme montado. Os frames e o áudio da montagem são a entrega do filme.

## Continuação

1. Use um workflow nativo que já gere H3. Conecte seu conditioning positivo e latent de destino ao **H3 Continuity · Prepare**.
2. Conecte a saída completa do sampler anterior em `source_latent`. Para vídeo externo, use **Import Video**, ligue `source_images` e o VAE, e forneça o áudio da origem quando houver.
3. Leve as saídas `positive` e `latent` do Prepare ao guider/sampler. Leve `plan` ao Assemble.
4. Decodifique o resultado completo do sampler. Ligue imagens e áudio decodificados ao **Assemble**, junto com as imagens e o áudio da origem para devolver o filme completo.
5. Salve o latent completo do sampler com **Save Latent** e conecte também `plan` para manter o tempo absoluto da cadeia.

`source_latent` tem prioridade sobre imagens externas. A resolução latente da origem deve coincidir com a do destino. Imagens externas são normalizadas pelo Import; mudar resolução pela via de pixels exige uma nova codificação VAE.

| Método | Comportamento |
|---|---|
| `pinned_av` | Fixa o prefixo visual e as linhas completas de áudio que puderem ser alinhadas. |
| `pinned_prefix` | Fixa o prefixo visual e fornece contexto de áudio como guia. |
| `anchors` | Fornece contexto como keyframes; o overlap visual é regenerado. |

Compare `pinned_av`, `pinned_prefix` e `anchors` no seletor `method` do mesmo Prepare. Os workflows 01/02 usam feather ligado (17 frames, força 1); `anchors` regenera o overlap por guias e não aplica esse controle. Use as mesmas referências limpas de identidade, roupa e cenário em todas as etapas. O prompt deve descrever o estado atual, a ação seguinte e a trajetória da câmera; referências não substituem memória do estado de objetos fora do quadro.

Um destino de **124 frames**, com **22 frames** de contexto, entrega **102 frames novos**, ou 4,25 s. A duração do destino inclui o contexto. Escolher um seed pode alterar a costura; verifique em movimento e com som.

### Feather e handover

`feather_frames=0` conserva a borda fixa. Valores positivos liberam gradualmente parte do contexto visual; `feather_strength` controla a intensidade e `feather_curve` a curva. Isso modifica a geração, sem dissolvência entre duas poses.

O plano calcula `handover_frame` a partir da máscara efetiva: a montagem mantém a origem até a primeira célula liberada e utiliza daí em diante os frames reparados. **Ligue `source_images` ao Assemble para entregar o feather**. Sem essa conexão, Assemble rejeita uma montagem que descartaria o reparo.

Feather troca os frames da borda nas mesmas coordenadas da timeline; não desloca os tempos de áudio. O áudio original continua preservado até a emenda nominal. Se o reparo mudar gestos, impactos ou fala nessa borda, preservar esse áudio não garante correspondência com o movimento regenerado.

O suporte é quantizado à grade H3. Não interprete `feather_frames` como uma quantidade exata de frames RGB independentes. Compare primeiro feather curto e força moderada; aumentar o reparo permite mudanças maiores também na origem.

### Acumular o canvas AV completo

**H3 Continuity · Assemble Latents** junta etapas sem decode/encode VAE e permite fornecer o filme completo a retakes longos e ao Joint Refine:

1. `source_latent` recebe o canvas acumulado anterior, completo.
2. `sampled_latent` recebe a saída bruta completa do novo sampler, incluindo contexto.
3. `plan` recebe o mesmo plano do Continuity Prepare desta etapa.
4. Salve a saída LATENT do Assemble Latents e use-a como `source_latent` na próxima etapa. Repita para acumular a cadeia.

A montagem visual segue o handover: conserva a origem até a célula protegida e inclui a borda reparada do novo take. O resultado remove `noise_mask`; uma nova edição deve definir sua própria região. O áudio conserva todas as linhas da origem e acrescenta as linhas novas para a duração acumulada, com ajuste de até uma linha de 25 ms no corte do overlap, informado no relatório. A entrega de gravações originais continua pelo Assemble de imagens/áudio.

Se `source_latent` receber apenas a última janela anterior, o resultado conterá **essa janela mais a continuação**, não todo o filme. Para manter a cadeia inteira, salve e realimente o canvas acumulado desde a primeira etapa. Sua RAM e o arquivo de checkpoint crescem com a duração.

## Bridge para uma âncora futura

Use **H3 Bridge · Prepare** para gerar o intervalo entre o final de A e o começo de B. Conecte:

- `left_latent`: saída completa de A;
- `right_latent`: saída completa de B;
- `latent`: destino vazio H3 para a ponte;
- `positive`: prompt da transição e referências canônicas.

O destino precisa comportar os dois contextos e um meio livre. Com 192 frames de destino e 39 de contexto por lado, a ponte acrescenta **114 frames** entre A e B. Origem, destino e âncora precisam ter a mesma resolução latente.

Prepare fixa a cauda de A e a cabeça de B, deixa o meio livre e permite reparo opcional nas duas bordas. As saídas entram no sampler. **Bridge Assemble** recebe seu decode completo, `plan`, imagens completas de A e B e respectivos áudios. Remove contexto duplicado e entrega os reparos nas duas costuras.

Para reduzir a dependência recursiva, gere B primeiro como um clipe novo com referências limpas, planejado para o mesmo plano: personagem, cenário, enquadramento, posição, velocidade e direção de movimento compatíveis. Depois gere a ponte A → B. Assim B não é uma extensão direta do latent degradado de A. O exemplo conceitual é `A → B → C → ponte → E novo`.

Uma ponte pode rejeitar na prática um par incompatível: transformar uma pose ou câmera muito diferente em poucos frames pode causar aceleração, morphing ou mudança de composição. Mais contexto não garante uma solução; escolha outra âncora, seed ou duração. O mecanismo reduz uma fonte de acúmulo, sem garantir reset perfeito da qualidade nem ausência de deriva.

`audio_policy=preserve` fixa as linhas de contexto alinháveis; `generate` deixa o áudio latente da ponte livre. Na montagem, `left_audio` e `right_audio` restauram as faixas originais fora do meio. Se não os conectar, só haverá o som gerado disponível; ausências ficam silenciosas.

## Retake temporal e espacial

**H3 Retake · Prepare Mask** recebe `source_latent`, preferencialmente sem recodificação. Alternativamente, recebe `source_images` e o VAE: imagens externas são codificadas uma vez e preenchidas internamente até a grade H3; Assemble conserva a contagem original.

Conecte seu LATENT ao sampler normal, mantendo prompt/referências apropriados à cena. Decodifique tudo e ligue o decode, as imagens originais e `plan` ao **H3 Retake · Assemble**.

| Controle | Efeito |
|---|---|
| `start_frame`, `end_frame` | Intervalo do clipe fornecido, com fim exclusivo. Não são tempos absolutos do filme. |
| `offset_frames` | Desloca o intervalo. |
| `grow_frames` | Expande as duas bordas como halo de reparo; valores negativos contraem. |
| `feather_frames` | Reduz a força dentro das bordas; feather longo pode deixar pouco centro plenamente editável. |
| `curve` | `smoothstep` ou `linear`, também usado no fade do áudio editado. |
| `video_strength` | Força visual, padrão 1. Zero preserva todo o vídeo. |
| `mask` | MASK opcional: branco edita, preto protege, cinza reduz a força. |
| `edit_audio` | Libera o intervalo sonoro; padrão falso, independente da máscara espacial e da força visual. |

Uma MASK pode ter formato `[H,W]`, `[1,H,W]` ou `[source_frames,H,W]`. Para acompanhar rosto/objeto em movimento, forneça uma máscara por frame do clipe completo. O pack não realiza segmentação ou tracking automaticamente.

A máscara é reduzida por máximo sobre o suporte temporal e sobre os patches espaciais nativos **2×2 de latent**. Isso faz o sampler e o modelo concordarem sobre a região editável e expande o suporte para fora da seleção. Na escala habitual do VAE H3, cada patch cobre 32×32 pixels; o relatório mostra o intervalo temporal efetivo.

Assemble restaura os frames originais fora do suporte temporal. Com MASK, restaura também os pixels protegidos dentro do intervalo, utilizando o suporte espacial quantizado. Não aplica crossfade visual. Máscara totalmente preta ou força visual zero entregam a origem visual integralmente; `edit_audio=true` ainda pode editar o som.

Mesmo quando latentes ficam fixos, o decoder temporal pode influenciar pixels próximos. Por isso a montagem restaura a origem fora da região entregue. Em resolução diferente, a origem é redimensionada com crop central antes da preservação; não equivale a conservar o arquivo de pixels na resolução anterior. Examine as bordas espaciais e temporais no resultado.

Os workflows [11_retake_differential.json](../workflows/11_retake_differential.json) e [12_retake_spatial_differential.json](../workflows/12_retake_spatial_differential.json) mostram as conexões prontas para retake temporal e espacial.

**Differential Diffusion** permanece experimental. Conecte o retake LATENT exato e os mesmos SIGMAS do sampler. O adaptador sincroniza máscara do sampler e timesteps H3; não combine com outro node de máscara dinâmica nem com Joint Refine.

## Áudio

`source_audio` é o som da origem, começando no seu primeiro frame. `audio` do Assemble é o decode completo do sampler, incluindo overlap. `soundtrack` do Prepare de continuação é uma gravação que deve orientar a nova geração.

`soundtrack_mode=full_video` começa a gravação em zero do filme; `continuation_only` começa a gravação na emenda, mantendo o som anterior. `missing_audio=generate` libera os trechos sem gravação dentro da janela processada; conecte o decode de áudio para entregá-los. `silence` mantém preenchimento silencioso. Consulte o [guia de áudio original](AUDIO_ORIGINAL.md).

No retake, `edit_audio=false` entrega `source_audio` intacto na duração do clipe, ou silêncio se ausente. Com edição habilitada, conecte também o áudio gerado: Assemble mistura o intervalo com fades nas bordas. Preservação de amostras originais na montagem não garante lip sync ou continuidade de voz. MP4 pode recomprimir áudio; salve FLAC em paralelo para conservar a faixa sem perdas.

## Salvar e retomar

**Save/Load Latent** salva os dois streams completos em safetensors, sem converter seus dtypes. O caminho é relativo a `ComfyUI/output/`. Nomes existentes recebem sufixo; `plan` de continuação conserva `end_seconds`.

**Save/Load Conditioning** salva os tensores e os metadados de positive/negative, incluindo referências e keyframes presentes no conditioning. O arquivo usa safetensors com árvore de metadados JSON; não usa pickle. Passe conditioning público dos nodes, antes do processamento interno do sampler. Objetos executáveis, controles e hooks arbitrários não são serializáveis.

O campo `settings` aceita um objeto JSON, por exemplo:

```json
{"seed": 94001, "steps": 8, "sampler": "euler", "context_frames": 39, "model_file": "seu-checkpoint-H3.safetensors"}
```

Load devolve o JSON como STRING para consulta. Ele **não aplica automaticamente** seed, steps, sampler, modelo ou LoRAs. O arquivo também não inclui os pesos do modelo. Salve o workflow junto dos checkpoints. Para uma etapa posterior sem anchors temporais antigos, conserve as referências canônicas e prepare novo conditioning: restaurar keyframes de um take anterior pode impor posições incompatíveis.

## Joint Refine experimental

**H3 Joint Refine · Context Windows** recebe MODEL e um LATENT AV completo já existente. Conecte esse mesmo LATENT ao sampler e utilize a saída MODEL do node. Configure denoise de refino no sampler; o node não muda o denoise nem aumenta resolução sozinho.

Cada avaliação usa janelas do mesmo estado ruidoso e do mesmo sigma. As predições sobrepostas são ponderadas e normalizadas antes de o sampler avançar o canvas completo. As janelas começam em ciclos de cinco passos latentes; comprimentos seguem `5k+2`. Máscaras de vídeo/áudio são fatiadas, keyframes intersectantes são deslocados e referências globais são mantidas. Acúmulo FP32 em RAM reduz a necessidade de cópias do filme completo na GPU.

Use um único conjunto global de prompts/referências. Joint Refine não monta uma timeline; forneça o canvas acumulado pelo **Assemble Latents**. Também não tem tabela de conditioning por clipe, upscale aprendido ou passagem global dilatada. Rejeita outro context handler, Differential Diffusion dinâmico e conditioning por área/ControlNet. Uma troca de modelo/LoRA ou denoise alto pode mudar ações, composição e identidade; o draft não é um render determinístico em baixa resolução do resultado final.

`window_frames` é arredondado para uma janela H3 válida e `overlap_frames` para overlap compatível. Mesmo overlap solicitado zero pode resultar em cinco frames compartilhados por causa da grade. Mais janelas aumentam o trabalho por passo; janela menor limita contexto. A técnica deve ser avaliada separadamente do transporte e da montagem de continuidade.

## Workflows do canvas

Os [exemplos de continuação](../workflows/02_continuar_latent.json), [bridge](../workflows/05_bridge.json), [retake temporal](../workflows/06_retake.json), [retake espacial](../workflows/07_retake_spatial.json), [joint refine](../workflows/08_joint_refine.json) e [âncora futura](../workflows/09_future_anchor.json) abrem no canvas. Selecione seus modelos e checkpoints locais. As versões `.api.json` correspondentes destinam-se à API do ComfyUI.

## Fontes da abordagem

- [Retake oficial LTX](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/retake.py) e [extend do LTX-Desktop](https://github.com/Lightricks/LTX-Desktop/blob/main/backend/services/retake_pipeline/ltx_retake_pipeline.py): máscaras temporais e regeneração de um halo da costura. A capacidade aprendida do LTX não é transferida ao H3 por este pack.
- [Diff-VF](https://arxiv.org/abs/2608.05976): combina janelas ponderadas e percurso global temporal; inspira investigação de sampling conjunto, sem demonstrar desempenho deste node em H3.
- [FreeLong++](https://arxiv.org/abs/2507.00162): fusão multiescala em atenção, com testes em Wan2.1 e LTX-Video. Não está implementado aqui.
- [RIFLEx](https://arxiv.org/abs/2502.15894): modifica frequência RoPE para extrapolação de comprimento. Não é uma correção direta da deriva em cadeias de janelas nativas.
- [History-Guided Video Diffusion](https://arxiv.org/abs/2502.06764): depende de treinamento/fine-tuning apropriado, portanto não equivale a este método training-free.
