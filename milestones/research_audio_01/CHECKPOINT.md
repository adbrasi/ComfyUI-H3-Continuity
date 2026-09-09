# Áudio original — checkpoint 01

Dois modos no H3 Continuity Prepare: full_video substitui a trilha inteira; continuation_only mantém o som anterior e inicia a gravação fornecida na emenda. O plan transporta o áudio ao Assemble. Arquivos maiores são cortados; menores recebem silêncio. Sem soundtrack, a continuação aprovada permanece igual.

[Guia e conexões](../../docs/AUDIO_ORIGINAL.md). Workflows UI/API em ../../workflows/03_audio_full_video e 04_audio_continuation_only. Copie o WAV de demonstração daquela pasta para input/. Restaure h3_source_3s.mp4 com o script do checkpoint research_retake_01. Modelos/hashes são os do Marco 01.

67 testes de código aprovados. Dois renders completos na instalação principal: FL2VA INT8 + Turbo, Euler/simple, 8 steps, 736×416. Origem 72 frames/3 s; alvo 124 com 22 de contexto; entrega 174 frames/7,25 s. O áudio de demonstração tem 12 s e é composto por tons, permitindo conferir o instante em que entra. Workflows API executados; UI verificados estruturalmente.

No FLAC exportado, a gravação fornecida foi preservada com erro zero nos dois modos. O som da origem no modo de concatenação apresentou erro máximo 1,5259e-5 frente à decodificação float do MP4, compatível com meia unidade de quantização PCM 16-bit na exportação FLAC. Os testes de tensores verificam as amostras originais antes da exportação. MP4 pode comprimir o áudio.

Resultados numéricos e tempos em results.json. O primeiro render incluiu carregamento; não usar essa comparação como diferença de velocidade entre modos. Esta validação confirma condicionamento executável e montagem temporal/amostras; não mede qualidade de sincronização labial.

Os snapshots anteriores permanecem intactos. Este checkpoint também inclui a correção de redimensionamento automático do Assemble.
