# Memória e continuação de vídeo sem treino: aplicações possíveis ao H3

Pesquisa em fontes primárias em 2026-09-09. Escopo: MiniMax H3 first/last-frame, Turbo, 8 steps, RTX 5090. Este relatório não implementa patches, não instala dependências e não registra resultados novos de render. As propostas abaixo são hipóteses de engenharia, não capacidades já demonstradas no H3.

## Conclusão prática

O próximo experimento mais útil é comparar **cauda curta**, **cauda longa** e **cauda curta mais referência visual antiga**. Ele responde se vale comprar mais memória temporal ou se poucos frames escolhidos preservam a identidade por menos custo. A ideia de congelar latents passados já existe no modo `pinned_prefix`/`pinned_av` do pack; faltam ampliar o contexto e medir como isso afeta o futuro, especialmente quando uma entidade desaparece antes da emenda.

A interface atual oferece 5, 22, 39 e 56 frames. A função local `tail_length` aceita matematicamente valores maiores e ajusta à grade `5 + 17k`; 112 pedidos correspondem a 107 utilizados nesse esquema. Isso é uma restrição da interface/política atual, não prova de um limite fundamental de 56 frames do modelo. Auditar suporte do target, alinhamento temporal e memória é necessário antes de ampliar. Fontes locais: `custom_nodes/h3_continuity/nodes.py`, funções `tail_length` e `H3ContinuityPrepare`.

## Oito trabalhos selecionados

### 1. FreeMem — memória de ruído, tokens e atenção

**Training-free: sim, nos backbones avaliados.** Mantém três memórias: componentes de ruído, features entre blocos e pares key/value da atenção, indexados por camada e timestep. A memória KV é concatenada ao contexto atual e atualizada por similaridade. A avaliação usa VideoCrafter2 e StreamingT2V, além de experimentos com AnimateDiff; não apresenta H3. Na tabela principal, FreeNoise passa de 350 para 500 segundos e FreeLong de 370 para 520 segundos. Logo, ausência de treino não implica ausência de custo. [Paper AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/download/37783/41745).

**Aplicabilidade inferida:** é o desenho mais próximo de guardar aparência fora da janela recente. Portar tudo seria invasivo; um banco pequeno de features de frames escolhidos, em poucas camadas, seria um primeiro protótipo. Precisaria separar tokens vídeo/áudio/texto, preservar o nível de ruído e verificar posições. Risco: copiar aparência também pode impor pose antiga ou estagnar o movimento. Não encontrei um repositório oficial confirmado nas buscas realizadas.

### 2. Ouroboros-Diffusion — memória específica do sujeito

**Training-free: sim.** Parte de FIFO-Diffusion, com frames em diferentes níveis de ruído. Combina inicialização de latents, atenção entre frames orientada ao sujeito e orientação recorrente. Extrai regiões de sujeito por mapas de cross-attention e mantém um banco de features; a orientação de longo prazo inclui otimização do latent com gradiente. [ArXiv](https://arxiv.org/abs/2501.09019), [paper AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/download/32205/34360).

**Aplicabilidade inferida:** inspira guardar o cachorro, não somente um resumo global do cenário. A implementação integral é pouco adequada ao objetivo de rapidez: exige acesso a attention maps, passos com gradiente e uma agenda de ruído diferente. Não transplantar para H3 Turbo supondo que os oito passos toleram o mesmo mecanismo. Um recorte/âncora selecionado pelo usuário é uma aproximação mais barata para testar se memória do sujeito ajuda antes de implementar SACFA.

### 3. FreeLong / FreeLong++ — contexto global e detalhe local

**Training-free: sim.** FreeLong mistura baixas frequências das features globais com altas frequências das locais. FreeLong++ usa várias escalas temporais e testa Wan2.1-1.3B e LTX-Video; também usa amostragem esparsa de keys na ramificação global. A tabela de ablação relata 50 segundos para global, 96 para FreeLong++ e 74 para a variante esparsa, no experimento descrito. Esses números não estimam o custo no H3. [Paper FreeLong, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ed67dff7cb96e7e86c4d91c0d5db49bb-Abstract-Conference.html), [FreeLong++ completo](https://arxiv.org/html/2507.00162v1), [código dos autores](https://github.com/aniki-ly/FreeLong).

**Aplicabilidade inferida:** bom fundamento para preservar contexto distante sem tratar cada interação temporal com o mesmo detalhe. É intervenção em features/atenção, não simplesmente dar resize no vídeo. H3 mistura modalidades em um layout; uma janela aplicada ingenuamente por índice de token pode cortar áudio, texto ou referências. Prioridade posterior ao teste com âncoras nativas.

### 4. FreeNoise — correlação no ruído e janelas

**Training-free: sim.** Reorganiza ruído inicial para criar correlação de longo alcance e usa atenção temporal por janelas; também suporta vários prompts. Os autores relatam overhead de aproximadamente 17% na configuração estudada. [Paper](https://arxiv.org/abs/2310.15169).

**Aplicabilidade inferida:** pode inspirar continuidade de textura e reduzir variações de estilo sem outro modelo. Porém ruído correlacionado não armazena, por si, que o cachorro tinha certa marca no focinho. Copiar ruído de maneira forte também pode produzir repetição. Uma ablação de inicialização apenas é pequena, mas só faz sentido após guardar explicitamente ruído/seeds do trecho anterior; reutilizar latents limpos como se fossem ruído é outro procedimento e muda a distribuição.

### 5. RIFLEx — extrapolar duração mexendo no RoPE

**Training-free: sim para o resultado de 2×; a variante de 3× usa fine-tuning.** Reduz uma frequência intrínseca do posicionamento rotativo para diminuir repetição e preservar movimento ao extrapolar comprimento. [Paper](https://arxiv.org/abs/2502.15894), [projeto dos autores](https://riflex-video.github.io/).

**Aplicabilidade inferida:** interessante caso aumentar a duração total produza repetição/movimento lento. Não é memória de identidade e não é pré-requisito para subir um limite arbitrário da UI. H3 tem RoPE e coordenadas de modalidades; copiar a mesma constante de Wan/Hunyuan sem análise não é correto. Primeiro medir contexto longo dentro de durações já usuais, depois considerar extrapolação posicional.

### 6. MemRoPE — memória comprimida com posições desacopladas

**Training-free: sim nos modelos autoregressivos SelfForcing/LongLive avaliados.** Mantém streams de memória de longo e curto prazo por médias móveis; guarda keys sem rotação e aplica RoPE no momento da atenção. Isso evita misturar fases posicionais incompatíveis ao agregar histórico. [Projeto dos autores](https://memrope.github.io/), [paper](https://arxiv.org/abs/2603.12513).

**Aplicabilidade inferida:** a lição útil é separar conteúdo da posição antes de comprimir memória. Não equivale a ligar KV cache convencional num transformer bidirecional H3: features do prefixo podem depender do restante do contexto e do timestep. Cache seria uma aproximação a validar, não reutilização exata. A média também pode diluir um objeto que apareceu brevemente; retrieval de frames específicos pode servir melhor ao teste do cachorro.

### 7. FramePack — histórico antigo comprimido e recente detalhado

**Training-free: não como adaptação completa.** Propõe estrutura de rede que comprime frames por importância e mantém o orçamento de contexto estável, com estratégia contra drift. Modelos existentes são ajustados para essa estrutura. [Paper](https://arxiv.org/abs/2504.12626), [projeto dos autores](https://lllyasviel.github.io/frame_pack_gitpage/).

**Aplicabilidade inferida:** responde diretamente à ideia de mandar mais histórico com resolução menor. O princípio é excelente, mas o H3 precisa entender os tokens e sua geometria. Reencodar frames pequenos e ampliá-los para a mesma grade não reduz os tokens vistos pelo DiT. Uma referência nativa com grade realmente menor pode reduzir custo se o layout aceitar; verificar suporte e qualidade seria um experimento próprio, sem chamar isso de FramePack implementado.

### 8. StreamingT2V — memória recente e aparência antiga separadas

**Training-free: não para os módulos principais.** Usa CAM para condicionar com os oito frames recentes e APM para aparência de longo prazo de uma âncora. O estágio de refinamento com blending aleatório dispensa treino adicional; isso não torna CAM/APM training-free. [Paper CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Henschel_StreamingT2V_Consistent_Dynamic_and_Extendable_Long_Video_Generation_from_Text_CVPR_2025_paper.pdf), [ArXiv](https://arxiv.org/abs/2403.14773).

**Aplicabilidade inferida:** a separação entre memória de movimento e memória de aparência é mais transferível que o código desses módulos. Podemos testar o mesmo princípio usando condicionamentos que o H3 já entende: cauda temporal recente mais frame antigo de referência, sem treinar novo adaptador. O sucesso precisa ser demonstrado no checkpoint first/last-frame, pois uma entrada aceita pelo código pode ter utilidade diferente entre checkpoints.

## Três experimentos pequenos, em ordem

### A. Mais cauda mantendo a tarefa futura comparável

Usar a mesma fonte aprovada, resolução e prompt de interação/câmera; comparar 22, 56 e 107 frames de contexto, todas com FL2VA + Turbo e 8 steps. Executar um seed para triagem e só repetir os dois melhores em outros seeds. Registrar frames realmente utilizados, segundos novos, tempo e pico de VRAM. Gerar o mesmo número de frames novos quando a grade permitir; quando não permitir, mostrar explicitamente a diferença e comparar o mesmo intervalo pós-emenda. Se for fixado o total para igualar custo, reconhecer que mudar contexto muda a duração futura.

Avaliar separadamente: emenda, velocidade e direção do movimento, pose, geometria e atendimento ao prompt. Prefixo preservado bit a bit demonstra conservação do passado; não demonstra que o futuro respeitou esse passado. Comparar sempre a região nova.

### B. Entidade sai e volta: teste causal de memória

Produzir uma fonte em que a entidade exista somente no início e fique ausente por pelo menos toda a cauda curta. Exemplo: cachorro alado azul, com orelhas de formatos diferentes e pequena marca clara em um lado do rosto. Evitar que os detalhes distintivos estejam no prompt de continuação; pedir apenas que o cachorro volte. Variantes: cauda curta; cauda longa que inclua a entidade; cauda curta mais frame antigo; cauda curta mais frame antigo trocado por outra aparência.

O controle com referência trocada é fundamental: um cachorro azul que aparece por causa do texto não prova memória. A referência deve alterar atributos não descritos, na direção esperada. Manter câmera em movimento e uma nova ação para penalizar soluções que congelam tudo. Comparação visual lado a lado/crops é o primeiro julgamento; embeddings podem ajudar depois, mas similaridade semântica alta não prova identidade exata. Não colar o objeto no resultado por pós-processamento.

### C. Memória visual escolhida versus descrição textual

Com a mesma fonte do teste B, comparar cauda + descrição escrita do passado, cauda + referência visual antiga e cauda + ambos. Texto funciona como resumo semântico: pode lembrar que houve um cachorro e descrever características, mas perde detalhes não verbalizados. Se a referência visual funcionar, testar uma versão de menor resolução que realmente reduza tokens, sem mudar a resolução da cauda.

Não concatenar embeddings CLIP/DINO arbitrários à saída do text encoder H3: forma tensorial compatível não significa representação compatível. A opção training-free simples é traduzir a observação para texto; a opção visual usa caminhos nativos. Um adaptador de embeddings visuais geralmente precisa ser treinado, salvo compatibilidade demonstrada. Retrieval por embeddings pode selecionar qual frame reenviar sem exigir que o gerador entenda o embedding.

## Ordem de implementação recomendada

1. Liberar contexto maior com contagem explícita e validar A, preservando o checkpoint estável.
2. Testar memória esparsa por referências nativas em B/C, sem outro modelo nem patch global.
3. Se o ganho justificar, investigar banco de tokens inspirado em FreeMem/Ouroboros, limitado por orçamento e separado por camada/timestep/modalidade.
4. Deixar compressão KV, atenção espectral e RoPE experimental para uma versão opcional. Não ativar de padrão enquanto não vencer o pack atual em continuidade, movimento e custo.

Nenhum desses papers oferece garantia geral de ausência de flicker, corte ou esquecimento. Bons resultados de consistência média também podem esconder pouca movimentação ou fracasso em reentrada de uma entidade. O benchmark específico acima precisa complementar a inspeção dos vídeos aprovados pelo usuário.
