# H3 Continuity

Continuação de vídeo MiniMax H3 com contexto em latent, máscaras nativas e montagem a 24 fps. Oito nodes, sem bibliotecas adicionais ou alterações globais no ComfyUI.

O modelo ainda pode errar mãos, objetos, identidade, movimento e sincronização labial. Este pack resolve o transporte e o alinhamento do contexto; não promete eliminar todos os artefatos da geração.

## RETAKE e feather experimental

O pack mantém o feather da continuação desligado por padrão (`feather_frames=0`) e acrescenta RETAKE temporal, montagem com extremos preservados e Differential Diffusion adaptado opcional. Veja [workflows, controles e resultados](../milestones/research_retake_01/CHECKPOINT.md). O node de Differential Diffusion modifica callbacks apenas no modelo clonado; não faz parte do caminho padrão.

## Marco aprovado

O vídeo com duas emendas e todo o estado necessário para retomá-lo estão em [Marco 01](../milestones/marco_01/CHECKPOINT.md). Inclui vídeo aprovado, três checkpoints AV, workflows do canvas/API, seeds e manifesto de reprodução. Use a tag `v0.1.0-marco-01` para voltar a este estado.

## Instalação

Copie a pasta `h3_continuity` para `ComfyUI/custom_nodes/` e reinicie o ComfyUI. Procure **H3 Continuity**. Requer ComfyUI com H3 nativo e máscaras AV corrigidas; testado em **0.34.0**. Não execute `pip install torch`: todas as dependências de execução já fazem parte do ComfyUI.

Os testes de desenvolvimento usam `pytest`; ele não é necessário para usar os nodes.

## Comece aqui

Abra `workflows/01_continuar_video.json`. Use seu vídeo em **Load Video**, descreva a continuação em **MiniMax H3 Image to Video** e execute. O workflow usa first/last frame, Turbo 8 steps, 736×416 e contexto de 22 frames. Os nomes dos modelos correspondem aos arquivos instalados nesta máquina.

- **Import Video**: converte o vídeo para 24 fps e o tamanho escolhido. Ligue suas imagens e áudio tanto ao Prepare quanto ao Assemble. Vídeos em outro FPS usam seleção/repetição de frames, sem interpolação artificial.
- **Prepare**: transporta o final do vídeo e cria o condicionamento/máscara. Suas saídas `positive` e `latent` entram no guider e no sampler. `plan` entra no Assemble e no Save Latent.
- **Assemble**: recebe imagens/áudio decodificados do sampler, remove o contexto repetido e acrescenta somente a parte nova. Com `source_images` e `source_audio`, devolve o vídeo completo preservando o original normalizado pelo Import.
- **Save Latent**: salva os dois streams do sampler em safetensors, sem conversão de precisão. Use o latent **completo, antes de qualquer recorte**. Arquivos existentes recebem outro número.
- **Load Latent**: carrega o checkpoint da pasta `output/` para a próxima continuação.

Use `workflows/02_continuar_latent.json` para continuar de um checkpoint. O latent leva o contexto de imagem e som; não é necessário recodificar o MP4. Para montar um filme com muitas etapas, salve cada trecho novo e monte-os depois: concatenar todos os frames em cada etapa faz o consumo de RAM crescer.

## Métodos

| Método | O que fica fixo | Uso |
|---|---|---|
| `pinned_av` | Prefixo de vídeo + linhas completas e alinhadas do áudio | Ponto de partida para continuidade de imagem e som |
| `pinned_prefix` | Prefixo de vídeo; áudio por guia temporal | Comparar quando a fixação do som prejudicar uma mudança sonora |
| `anchors` | Contexto fornecido como keyframes; o prefixo é regenerado | Comparação com condicionamento por guias |

A máscara fixa somente o começo; o restante continua livre para ações e câmera novas. Não usamos dissolvência, interpolação de frames ou correção automática de cor para disfarçar a emenda. A escolha final deve considerar o vídeo em movimento e o som, não apenas uma métrica.

O `audio_context_seconds` controla quanto áudio anterior é fornecido. `0` desliga esse contexto; não silencia o áudio novo. Quando `source_latent` está conectado, ele tem prioridade sobre `source_images` e `source_audio`.

## Duração e limites

H3 trabalha com `5 + 17k` frames a 24 fps. Um target de 124 frames com contexto de 22 entrega **102 frames novos = 4,25 segundos**. Com uma fonte de 3 segundos, o resultado montado dura 7,25 segundos. O número configurado no node nativo inclui o contexto: ele não é apenas a duração nova.

Um clipe externo de 72 frames é aceito integralmente na montagem, mas só sua cauda válida é codificada. Não descarte o final antes de conectá-lo ao Prepare. O latent anterior precisa ter a mesma resolução do target. Para mudar de resolução, entre pelo vídeo decodificado e pelo Import; isso exige uma recodificação VAE.

O vídeo tem uma grade temporal e o áudio outra, de 40 passos por segundo. Nem todo limite de frame coincide com um limite do áudio. O pack alinha as linhas à grade, fixa somente linhas completas anteriores à emenda e ajusta a duração do áudio na saída; ainda pode existir diferença subframe na reconstrução gerada.

Os anchors visuais que colidem com o prefixo são substituídos, com informação no relatório. Last-frame posterior, outros guias e referências são mantidos. Evite fornecer guias de áudio contraditórios na mesma janela.

## Usar uma gravação original

O `soundtrack` do Prepare aceita a **gravação completa da linha do tempo**, começando em zero. Conecte também o áudio VAE. O pack recorta a região correspondente ao vídeo novo e fixa esse áudio no target por máscara, permitindo que o vídeo seja gerado em relação a ele.

O Assemble usa as amostras da gravação original para o trecho novo, em vez da reconstrução do VAE. Para a parte anterior, ligue a gravação original em `source_audio` do Assemble. A gravação deve cobrir toda a duração do filme resultante. Uma saída MP4 pode comprimir o som; use **Save Audio** em paralelo para um arquivo FLAC sem perdas.

`source_end_seconds=0` calcula o ponto da emenda a partir da fonte ou do checkpoint. Se carregar somente a cauda de um filme maior, informe o instante absoluto onde ela termina. Em checkpoints sucessivos, conecte `plan` ao Save para preservar essa posição.

O áudio original é preservado na montagem, mas isso não garante dicção ou lip sync perfeitos no vídeo. Para simplesmente continuar o ambiente/voz gerados, deixe `soundtrack` desconectado.

## Prompts para continuação

Descreva primeiro o estado real no final do vídeo e o movimento que já está acontecendo. Depois peça a ação nova. Exemplo:

> A mulher continua caminhando no mesmo enquadramento lateral. Ela reduz o passo, para e abre a bolsa. A câmera faz um arco suave em direção à frente enquanto ela retira uma chave. Plano contínuo, iluminação constante. O som dos passos desacelera e dá lugar ao ruído do couro e da chave.

Não é preciso pedir que tudo fique estático. Uma mudança física gradual de câmera é compatível com continuação; uma troca instantânea de enquadramento no prompt pode induzir um corte.

## Verificação

Relatório de pesquisa: `docs/PESQUISA.md`. Resultados locais: `docs/TESTES.md`. Os MP4 completos, áudios sem recorte, checkpoints e folhas de contato ficam em `/workspace/H3-testes` nesta máquina.

Código de teste: `tests/test_nodes.py`. No Python do ComfyUI: `python -m pytest custom_nodes/h3_continuity/tests -q`.

## Áudio original

Use `soundtrack` no Prepare: `full_video` substitui toda a trilha; `continuation_only` mantém `source_audio` e começa sua gravação na emenda. O `plan` leva o áudio ao Assemble. Excedente é cortado; trechos sem gravação são gerados dentro da janela do modelo (`missing_audio=generate`). [Conexões e exemplos](AUDIO_ORIGINAL.md).
