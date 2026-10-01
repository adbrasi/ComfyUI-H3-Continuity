# Ação e aventura — 1 de outubro de 2026

Foram executados 12 grafos, com 21 chamadas de sampling, para testar movimento rápido, interação entre personagens, oclusões, câmera em arco e outra resolução. Todos terminaram com sucesso técnico. Isso não significa que anatomia, física e continuidade sejam perfeitas.

## Vídeos

Os arquivos estão em `/workspace/comfy/ComfyUI/output/h3_action_validation/`.

| Teste | Arquivo relativo | Duração |
|---|---|---:|
| Corrida na selva | `jungle/chain_jungle_hard_hard_00001_.mp4` | 17,92 s |
| Duelo, máscara fixa | `combat/chain_combat_hard_hard_00001_.mp4` | 17,92 s |
| Duelo, feather | `combat/chain_combat_feather_feather_00001_.mp4` | 17,92 s |
| Perseguição a cavalo na neve | `snow/chain_snow_hard_hard_00001_.mp4` | 17,92 s |
| Selva com bridge até âncora nova | `jungle/bridge_00002_.mp4` | 27,83 s |
| Retake de um trecho do duelo | `combat/retake_00002_.mp4` | 5,88 s |
| Duelo em 1024×576 | `combat_576p/chain_combat_576p_hard_00001_.mp4` | 13,67 s |
| Duelo fixo/feather lado a lado | `duelo_comparacao.mp4` | 17,92 s |
| Mosaico dos 12 testes | `todos_testes_4x3.mp4` | cerca de 28 s |

O mosaico é 4×3, em 2560×1152, sem áudio. Os vídeos começam simultaneamente. Cada painel mostra identificação e duração; ao terminar, conserva o último frame e mostra FIM. O retake é um trecho isolado, portanto seu relógio local não coincide com o filme inteiro. Os vídeos individuais incluem o áudio original de cada teste.

Ordem dos painéis: selva fonte/extensões/âncora/bridge; duelo fonte/fixo/feather/retake; neve fonte/extensões e duelo 1024×576 fonte/extensões. As fontes e a âncora são controles, não testes de extensão.

## Configuração

RTX 5090 de 32 GB, H3 FL2VA INT8, encoder Qwen3VL NVFP4/AWQ e LoRA Turbo de 8 steps já instalada. Euler/simple, 24 fps. As cadeias usam 39 frames de contexto AV, destino de 141 frames e referência visual persistente extraída da fonte. Cada extensão acrescenta 102 frames; três extensões mais fonte de 124 frames produzem 430 frames.

Selva, duelo e neve usam 736×416. O duelo adicional usa 1024×576 e duas extensões. Alterar resolução modifica o ruído latente e as trajetórias: o teste de maior resolução não é a mesma ação renderizada novamente com mais pixels.

O duelo compara máscara fixa com feather de 17 frames e força 1,0, usando a mesma fonte, prompts, seeds e scheduler. As trajetórias podem divergir depois do reparo porque a geração e o contexto seguinte mudam. O bridge usa um canvas acumulado de 430 frames e âncora nova de 124 frames; acrescenta 114 frames, totalizando 668. O retake trabalha na segunda janela bruta de 141 frames, em torno de uma transição, com áudio conservado.

Os prompts, seeds e grafos estão em [api/](api/) e [scenarios.json](scenarios.json); respostas da API em [history/](history/). [results.json](results.json) contém tempos e hashes. O [workflow do duelo feather](duelo_feather_workflow.json) pode ser aberto no canvas; requer o checkpoint de fonte salvo no caminho indicado, produzido por `api/combat_source.api.json`.

## Observações

- **Selva:** a personagem e o figurino amarelo persistem nas imagens amostradas. Folhagem e troncos passam na frente dela, elevando a diferença entre frames; essas oclusões também dificultam avaliar pernas, contato com o chão e identidade facial. A solicitação de saltos não está comprovada por uma avaliação semântica.
- **Duelo:** os dois figurinos e o pátio persistem, mas a câmera recua gradualmente e a coreografia muda. O usuário avaliou o vídeo com feather como “muito bom” ao assisti-lo; isso é feedback qualitativo, não uma medida automática de identidade ou física das espadas.
- **Neve:** nas amostras, cavalo, cavaleira e cachecol vermelho reaparecem após a oclusão por uma árvore. Não foi realizada análise biomecânica do galope ou rastreamento de identidade.
- **Bridge da selva:** mantém a personagem caminhando/correndo, com mudanças de folhagem e enquadramento perto do lado novo. Seus joins nominais são frames 430 e 544; as trocas efetivas de pixels após feather são 413 e 561.
- **Retake do duelo:** executou mantendo regiões protegidas; as amostras nas bordas não mostram uma grande descontinuidade. A habilidade de impor uma coreografia específica ainda exige revisão do vídeo.
- **Duelo 1024×576:** conserva os figurinos nas amostras, mas muda ponto de vista/coreografia; o último segmento apresenta mudança importante no indicador de luminosidade da região superior. Maior resolução não remove automaticamente deriva de composição.

Nas cadeias de 430 frames, as emendas nominais ficam em **5,17 s, 9,42 s e 13,67 s**. Com feather de 17 frames, as trocas efetivas de montagem começam antes: **4,46 s, 8,71 s e 12,96 s**.

## Métricas e limites

Diferenças RGB na emenda, em escala 0–255:

| Cadeia | Frame 124 | Frame 226 | Frame 328 |
|---|---:|---:|---:|
| Selva | 21,11 | 17,91 | 22,81 |
| Duelo fixo | 7,26 | 9,38 | 7,07 |
| Duelo feather | 7,22 | 9,71 | 7,39 |
| Neve | 20,99 | 19,86 | 21,61 |

Movimento rápido e paralaxe produzem diferenças altas mesmo com continuidade. Usamos também fluxo óptico Farneback em meia resolução e resíduo RGB após warping, comparando a emenda com três pares vizinhos de cada lado. O estimador pode falhar em oclusões, blur e superfícies sem textura; não certifica física, identidade nem ausência de cortes perceptíveis.

No duelo, o resíduo após warping foi menor com feather, mas a diferença RGB não mostrou vantagem uniforme. A aparência deve ser julgada no vídeo: não reduzimos o resultado a uma nota de qualidade. Não há crossfade na montagem visual; feather atua no sampling e entrega os frames reparados.

A variância do Laplaciano e a luminosidade do topo da imagem são indicadores descritivos. Em ambientes diferentes, o topo pode ser vegetação, céu ou paredes; mudanças de composição influenciam esses valores. Os relatórios completos e imagens estão em [metrics/](metrics/) e [previews/](previews/).

Esses testes ampliam a validação para ação, mas não estabelecem continuação ilimitada sem degradação. Permanecem limites de composição, física, memória de cenário e controle da coreografia.
