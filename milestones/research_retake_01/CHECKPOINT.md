# RETAKE e feather — pesquisa 01

Branch `research/retake`, derivada de `research/memory-context`. A instalação principal e os snapshots aprovados permanecem intactos. Sem novas dependências, pesos ou alterações no core do ComfyUI.

## Continuação: o padrão continua igual

Prepare ganhou três controles opcionais, acrescentados ao fim dos widgets para preservar a ordem anterior:

- `feather_frames = 0`: mantém o prefixo totalmente fixo, como no comportamento aprovado.
- `feather_strength`: limita a liberdade dada à borda; só atua quando feather_frames é maior que zero.
- `feather_curve`: linear ou smoothstep.

Com feather ativado, apenas os últimos frames do contexto ganham uma rampa de máscara. O restante do prefixo fica em zero; a região nova continua em um. A largura é avaliada na grade temporal dos latents. O áudio mantém a política do modo escolhido. Anchors não usa máscara de prefixo, portanto esses controles só têm efeito em pinned_prefix/pinned_av.

A montagem continua usando os frames da fonte antes da emenda, sem crossfade de vídeo. Permitir que o prefixo interno se altere pode melhorar ou piorar o encaixe com essa fonte. Não substituímos o modo campeão pelo feather.

## RETAKE

`H3 Retake · Prepare Mask` recebe um latent H3 bruto ou imagens a 24 fps com o VAE. O workflow para vídeo externo usa Load Video → Import Video → Retake Prepare. Ele mantém o começo e o fim e libera um intervalo temporal central. O prompt deve descrever a ação desejada nesse intervalo e seu retorno ao estado final conhecido.

Controles:

| Entrada | Efeito |
|---|---|
| start_frame / end_frame | Intervalo local ao clipe, com fim exclusivo; 24 fps |
| offset_frames | Desloca o intervalo inteiro |
| grow_frames | Expande ambos os lados; negativo contrai |
| feather_frames | Suaviza para dentro de cada borda; 0 deixa máscara dura |
| curve | Linear ou smoothstep |
| edit_audio | Desligado por padrão; preserva o áudio |

O feather do RETAKE tem padrão 8 para facilitar a experimentação; o da continuação tem padrão 0. Se o feather ocupar todo o intervalo, a força máxima de edição pode ficar abaixo de 1. O node informa o intervalo solicitado e o suporte efetivo dos latents: alguns representam 1 frame, outros 4, então a seleção nem sempre corresponde a um corte exato por pixel-frame. O preview é uma linha do tempo da máscara, não uma máscara espacial.

Vídeos externos fora da grade são preenchidos internamente repetindo o último frame, em vez de encurtar a entrega. `H3 Retake · Assemble` restaura os frames originais fora do suporte efetivo editado e devolve a duração original. A identidade dos pixels preservados vale **antes da compressão do arquivo final**; MP4 pode recomprimir. A reconstrução VAE também pode ter influência temporal além do suporte nominal, por isso a montagem e as duas emendas precisam ser avaliadas em movimento.

O latent bruto de uma entrada externa preenchida ainda inclui o padding. Para continuar ou editar novamente a entrega com sua duração exata, use o vídeo montado como entrada; não trate esse latent preenchido como se tivesse o número original de frames.

Conecte source_audio ao Assemble para conservar as amostras originais. Se edit_audio estiver ligado, ele usa o áudio gerado no intervalo e faz fades nas bordas; essa opção tem testes de estrutura/amostras, mas os renders deste checkpoint focam vídeo com áudio preservado. Sem source_audio, a saída preservada é silêncio. Para condicionar o modelo com o som real de um vídeo externo, conecte também source_audio e audio_vae ao Prepare.

## Differential Diffusion

O [paper original](https://differential-diffusion.github.io/) usa um mapa para controlar a quantidade de mudança por região durante a inferência. Feather é a forma suave do mapa. Differential Diffusion altera como esse mapa é liberado ao longo dos passos; não é sinônimo de blur.

Nesta versão do ComfyUI, o node genérico modifica a máscara usada pelo sampler, enquanto o H3 prepara suas máscaras por token em extra_conds antes da amostragem. Com uma máscara dinâmica, os dois caminhos precisam acompanhar o mesmo valor. A análise coincide com a preocupação documentada pelo [Context Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop/blob/main/docs/V0_5_ARCHITECTURE.md) para seus próprios modos dinâmicos; não copiamos seu algoritmo de Drift-Control.

`H3 Retake · Differential Diffusion (experimental)` instala callbacks apenas no ModelPatcher clonado. Ambos calculam a mesma máscara a partir de sigma/schedule; o wrapper usa a conversão nativa do H3 para atualizar as máscaras de vídeo e áudio. Não há estado global nem patch do core. Valores 0 e 1 continuam protegidos/livres, e strength 0 é identidade. Conecte o **mesmo latent de máscara e os mesmos sigmas do sampler**. Não combine com outro node de máscara dinâmica. A primeira validação GPU usa BasicGuider/Euler; outros samplers e combinações de wrappers não foram avaliados perceptualmente.

Em 8 steps há poucos estágios de liberação. Não há evidência, nestes testes, de superioridade consistente sobre o feather nativo. O node fica opcional e não é usado nos workflows padrão de continuação.

## Validação

56 testes de código passaram, incluindo preservação dos extremos, offset/grow/feather, vídeo externo com padding, duração de entrega, áudio original, compatibilidade do feather 0 e igualdade entre a máscara dinâmica recebida pelo sampler e a recebida pelo H3.

Renders com FL2VA INT8 ConvRot, Turbo FL2V8step v1.0, 8 steps, Euler/simple, 736×416. RETAKE em uma fonte de 124 frames: editar [34,90), pedindo que a mulher olhe, sorria e acene, depois retorne à caminhada. Foram comparados máscara dura, feather 8, feather 17 e feather 8 com Differential Diffusion. Todos produziram o gesto na região central nos frames inspecionados; não declaramos um vencedor de suavidade.

Nos quatro RETAKEs de latent bruto, o erro máximo das partes protegidas de vídeo foi 4,77e-7 no começo e no fim; áudio 1,19e-7. Isso verifica conservação numérica, não perfeição do movimento. Os renders com modelos já carregados levaram cerca de 20 s. O primeiro levou 39 s, incluindo inicialização; não comparar esses tempos como velocidade entre métodos.

A continuação aprovada da bicicleta foi repetida com a mesma seed 90123 e três opções: feather 8/força 0,5; feather 8/força 1; feather 17/força 0,5. Mudanças no gesto e nos detalhes são observações destas amostras, não prova de que feather causa ou corrige um erro específico. Baseline continua sendo o snapshot aprovado com feather 0.

Há também um teste externo de 72 frames, editando [22,56), com padding interno para 73 e entrega de 72 frames. Resultados completos e tempos estão em results.json; vídeos, máscaras e folhas de contato estão neste checkpoint e em `/workspace/H3-testes`.

## Usar

Instale esta branch como o pack h3_continuity e reinicie ComfyUI. Não instale duas cópias com os mesmos IDs de nodes no mesmo custom_nodes. A instalação principal desta máquina não foi trocada automaticamente.

Os exemplos UI/API estão em ui/ e api/. Para os inputs exatos da demonstração, execute `python milestones/research_retake_01/restore_inputs.py --comfy-root /caminho/ComfyUI`. Os modelos/hashes são os do Marco 01. Os workflows API foram executados; os UI foram gerados deles e verificados estruturalmente, sem uso manual em outra máquina.
