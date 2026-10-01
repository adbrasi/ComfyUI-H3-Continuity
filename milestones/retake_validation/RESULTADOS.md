# Retake com mudança evidente — espadas de energia e portal

Comparativo do duelo completo, 430 frames / 17,92 s:
`/workspace/comfy/ComfyUI/output/h3_retake_validation/comparacao_retake_duelo_completo_2x2.mp4`

Comparativo apenas do trecho, 141 frames / 5,88 s:
`/workspace/comfy/ComfyUI/output/h3_retake_validation/comparacao_retake_trecho_2x2.mp4`

| Posição | Abordagem |
|---|---|
| Superior esquerda | Original |
| Superior direita | Retake com máscara temporal suave |
| Inferior esquerda | Temporal + Differential Diffusion |
| Inferior direita | Temporal e espacial + Differential Diffusion |

Os três retakes usam exatamente o mesmo latent (`chain_combat_feather_02.safetensors`), prompt, seed 58702, Euler/simple, oito passos, Turbo LoRA 1, força visual 1. Feather ligado com oito frames nas duas bordas; áudio protegido. A máscara espacial é um retângulo central de 544×304 em x96/y112, expandido para os patches nativos do modelo. Não há versões feather-off.

Mudança pedida: espadas de energia ciano/magenta e portal laranja temporário. O prompt pede retorno às espadas normais ao final. O intervalo solicitado é [34,107), em coordenadas do latent local. Com a rampa e o grid temporal, o suporte entregue é [34,107). No duelo completo, o frame zero deste latent corresponde ao frame187; somente [221,294), de 9,21 a 12,25 segundos, é substituído. O resto vem do vídeo aprovado pelo usuário. Não há crossfade de vídeo na montagem. O MP4 completo passa por nova compressão H.264, portanto não se afirma igualdade byte a byte dos pixels exportados.

Os três jobs terminaram com status success. Os dois mosaicos foram verificados com ffprobe; vídeos completos possuem 430 frames e áudio original copiado do duelo aprovado. Os retakes curtos também conservam o mesmo áudio: os três FLACs decodificados para PCM float32 têm hash idêntico, registrado em audio_hashes.json.

O frame de inspeção aos 10,5 s do filme mostra portal e espadas coloridas nos modos temporais. No modo espacial, as espadas mudam e o portal não aparece nesse frame. A máscara limita a área disponível e serve para comparar preservação espacial, sem garantir que todo elemento do prompt será gerado. Temporal e Differential Diffusion parecem próximos neste exemplo; não concluímos superioridade geral.

As transições e a qualidade do efeito devem ser avaliadas em movimento. Preservar o contexto e restaurar o exterior da máscara não garante uma transformação fisicamente perfeita. Históricos, APIs, workflow importável de Differential Diffusion e scripts de montagem estão arquivados nesta pasta.
