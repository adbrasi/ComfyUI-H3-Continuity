# Validação — 1 de outubro de 2026

A versão 0.2.0 implementa continuação latente, reparo das bordas, bridge de duas pontas e retake temporal/espacial. A implementação foi instalada no ComfyUI local. Os testes demonstram funcionamento e preservação das regiões protegidas; não demonstram ausência universal de degradação.

## Ambiente e reprodução

- RTX 5090, 32 GB; ComfyUI `2d6b73283af2447bdd065ece4090b8c6b1784544`.
- Base do pack: `9a4cc930345a4805d56e0c901b7cc61b6f7a1268`.
- Vídeo 736×416, 24 fps; sampler Euler, scheduler simple, 8 steps.
- H3 `minimax_h3_fl2va_pruned_int8_convrot.safetensors` com LoRA Turbo 8-step já instalada. Nenhum peso foi treinado ou modificado pelo pack.
- Encoder Qwen3VL NVFP4/AWQ; VAE H3 vídeo FP16 e áudio FP32.
- Cena de controle: mulher de casaco verde caminhando lateralmente junto a uma parede de tijolos, câmera acompanhando e som ambiente/passos.

[results.json](results.json) registra prompt IDs, duração, arquivos e hashes. [executed/](executed/) contém os grafos efetivamente executados; [history/](history/) contém as respostas da API. Os caminhos desses grafos são da máquina de validação. Copie [inputs/](inputs/) para `ComfyUI/input/` quando reproduzir. Os pesos não estão incluídos.

Para os workflows de uso normal, abra os JSON do [canvas](../../workflows/05_bridge.json) em `workflows/`; `.api.json` é para API. O [guia](../../docs/continuation-retake.md) explica as conexões. Os scripts `validate_gpu.py` e `gpu_workflows.py` reproduzem submissões e métricas; `api_to_workflow.py` converte grafos API usando `/object_info`.

## O que foi verificado

**190 testes CPU passaram**, cobrindo máscaras, handover, contagem de frames, áudio em diferentes fases da grade, checkpoints e fusão de janelas. Foram 17 grafos GPU concluídos, totalizando 31 chamadas de sampling. As execuções estão individualmente registradas em `results.json`. Fora da instalação em custom_nodes, execute os testes com `COMFYUI_ROOT=/caminho/ComfyUI python -m pytest -q`.

| Caso | Resultado |
|---|---|
| Continuação fixa, feather e referência | Cada montagem: 226 frames / 9,42 s. |
| Duas cadeias com oito extensões | Cada filme: 940 frames / 39,17 s; mesmos seeds, com e sem referência canônica. |
| Bridge fixo e com feather | 379 frames, usando um segundo clipe de 141 frames. |
| Bridge para clipe novo independente | 362 frames / 15,08 s; 114 frames gerados entre dois clipes de 124 frames. |
| Bridge após oito extensões | 1178 frames / 49,08 s; canvas acumulado de 940 frames conectado ao clipe novo. |
| Retake temporal e espacial | 124 frames; a edição faz a personagem olhar para a câmera e acenar no intervalo central. |
| Joint Refine | Executado em canvases de 124 e 226 frames, com duas e três janelas respectivamente. |
| Assemble Latents | Canvas de 226 frames: 67 passos de vídeo e 377 linhas de áudio. |

As regiões latentes congeladas tiveram erro máximo de aproximadamente `4,77e-7` no vídeo e `1,19e-7` no áudio após sampling nativo. É tolerância de ponto flutuante, não identidade bit a bit. Na montagem latente, os tensores originais de vídeo e áudio permaneceram **bit a bit iguais** no trecho conservado. No Joint de 226 frames, o áudio protegido variou no máximo `7,45e-9`.

No retake espacial, 81,45% dos valores latentes visuais ficaram protegidos neste teste. Assemble restaura os pixels originais fora do suporte editável; testes de tensores RGB verificam essa preservação. Comparar dois MP4 comprimidos não prova identidade dos pixels antes da codificação. Consulte [latent_checks.json](metrics/latent_checks.json) e [assembly_joint.json](metrics/assembly_joint.json).

## Degradação e emendas

A referência canônica ajudou a conservar textura nos trechos finais desta cena. A média de variância do Laplaciano nos quatro últimos segmentos ficou aproximadamente em 124–129 sem referência e 161–172 com referência. Isso é apenas um indicador de nitidez, influenciado por enquadramento, textura, movimento e compressão. Não equivale a uma medida de identidade ou a comprovação de recuperação de qualidade.

As diferenças RGB entre frames nas emendas ficaram aproximadamente em 7,24–10,97/255 sem referência e 7,95–10,85/255 com referência. Algumas emendas com referência tiveram diferenças maiores. A referência portanto não resolveu automaticamente todos os aspectos da transição.

No teste curto, a emenda com máscara fixa teve MAE 8,04/255, contra aproximadamente 8,3 com feather. O feather **não mostrou vantagem suficiente para virar padrão**. A correção implementada garante que um reparo liberado pela máscara seja entregue, evitando o descarte que existia na montagem anterior.

O bridge para uma âncora nova manteve a caminhada e o enquadramento na inspeção das imagens amostradas, com mudanças sutis na textura da parede. As fronteiras nominais ficaram em 124 e 238, com MAE 8,25 e 8,44. Com feather de 17 frames, as trocas efetivas entre pixels originais e gerados ficam em 107 e 255. Não há dissolvência entre poses: o modelo gera o intervalo. Inspeção por contact sheet e MAE não certificam suavidade de movimento em todos os frames.

O Joint Refine provou execução das janelas no mesmo estado e sigma, com máscaras preservadas. **Não foi demonstrado que elimine a degradação**, nem que uma troca de LoRA/resolução conserve exatamente ações e detalhes. Permanece experimental.

O teste final conecta a cadeia sem referência, já com textura menos nítida, à âncora nova. A inspeção das [imagens finais](previews/bridge_after8.jpg) mostra a caminhada continuando e a textura mudando gradualmente em direção à âncora. O clipe novo fornece um destino de qualidade independente; a transição ainda pode alterar composição e detalhes do cenário. Isso valida a construção do bridge após uma cadeia longa, não uma garantia de correção universal.

Nesse bridge longo, força de feather 0,5 deixou uma mudança visível de luminosidade no início nominal da âncora, frame 1054: MAE 21,00/255, queda média RGB de aproximadamente 9–13 níveis. Repetir o mesmo seed com força 1,0 reduziu a MAE para 10,34 e a variação média para aproximadamente 1 nível. Os limites efetivos de montagem ficaram em MAE 8,24 e 7,73. Veja [antes](previews/bridge_bad_seam.jpg), [depois](previews/bridge_repaired_seam.jpg) e as [métricas](metrics/bridge_after8_repair.json). O exemplo de bridge usa força 1,0; esse resultado não transforma feather em padrão recomendado para toda continuação.

## Configuração recomendada

Comece com `pinned_av`, 39 frames de contexto e destino de 141 frames: cada extensão acrescenta 102 frames. Mantenha as referências canônicas e salve latentes/conditioning. Use feather zero como controle. Ative reparo curto quando houver uma costura problemática e examine os frames substituídos.

Para evitar depender sempre do próprio resultado, planeje uma âncora futura nova e compatível, depois gere o bridge até ela. Isso interrompe a herança recursiva naquela âncora; não garante que qualquer par de clipes possa ser unido naturalmente. Use retake para refazer regiões defeituosas, com um halo temporal e uma MASK que acompanhe o objeto quando necessário.

A validação cobre uma cena, uma resolução e um conjunto de pesos/sampler. Diálogo, oclusões longas, interações complexas, cenas novas e H3 sem Turbo precisam de avaliação própria. Não apresentamos “continuação infinita sem degradação” como resultado estabelecido.
