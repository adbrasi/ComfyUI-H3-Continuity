# Usar seu áudio na continuação

No Prepare, `soundtrack` é o arquivo que deve orientar a geração e aparecer no vídeo final. Conecte também o VAE de áudio H3 em `audio_vae`, e leve `plan` ao Assemble. A saída AUDIO do Assemble vai ao Create Video junto com a saída IMAGE.

| Objetivo | soundtrack_mode | Conexões |
|---|---|---|
| Substituir todo o som pelo seu arquivo | full_video | Load Audio → soundtrack do Prepare. O áudio da origem é substituído automaticamente no Assemble. |
| Manter o som anterior e acrescentar seu arquivo | continuation_only | Load Audio → soundtrack; áudio do vídeo de origem → source_audio do Prepare. O plan também transporta esse áudio ao Assemble; conectá-lo diretamente ao source_audio do Assemble continua válido. |
| Manter o som anterior e gerar som novo com H3 | Sem soundtrack | Áudio de origem → source_audio do Prepare e do Assemble, como antes. Áudio decodificado do sampler → audio do Assemble. |

`source_audio` representa o som **do vídeo de origem**. Não é uma entrada de trilha para o resultado inteiro. `audio` do Assemble representa o áudio gerado/decodificado da janela completa, incluindo o contexto; por isso não conecte um arquivo de continuação diretamente ali.

Exemplo: origem de 3 s e continuação de 4,25 s, resultado com 7,25 s.

- full_video: usa de 0 até 7,25 s do arquivo. O modelo recebe o trecho alinhado à sua janela temporal, inclusive o contexto.
- continuation_only: mantém os 3 s originais e acrescenta de 0 até 4,25 s do novo arquivo. O modelo recebe o contexto sonoro anterior e, após a emenda, a nova gravação.

Áudio excedente é cortado. Se o arquivo acabar antes, completa com silêncio: não estica, repete ou inventa som. Sem source_audio, o Assemble não consegue recuperar as amostras originais só pelo latent; forneça o áudio do vídeo. Na geração, o contexto sonoro do latent ainda é aproveitado quando disponível.

O Prepare codifica a gravação com o VAE de áudio e fixa os tokens de áudio enquanto gera o vídeo. O Assemble usa as amostras fornecidas, evitando a perda do encode/decode VAE na trilha final. Isso condiciona movimentos ao som, mas não garante sincronização labial perfeita. Arquivos MP4 podem comprimir o áudio na exportação. Em continuation_only, taxas de amostragem/canais diferentes são compatibilizados para concatenar.

Em cadeias, full_video começa no tempo zero da linha do tempo original. Preserve plan → Save e os metadados do latent, ou ajuste source_end_seconds ao usar um trecho recortado. continuation_only sempre começa o arquivo novo na emenda desta chamada. Sem source_images no Assemble, a saída é somente a continuação, com o trecho de áudio correspondente.

## Exemplos

`workflows/03_audio_full_video.json` e `04_audio_continuation_only.json`, com versões API. Nesta máquina estão em H3 Continuity → Audio original. Troque o WAV de tons de demonstração pelo seu áudio. Para reproduzir em outra máquina, copie `workflows/h3_soundtrack_demo.wav` para `ComfyUI/input/` e restaure o vídeo de origem dos checkpoints anteriores.

A partir deste ajuste, full_video substitui também o áudio da origem no Assemble. Anteriormente soundtrack só entregava o trecho novo; os snapshots anteriores conservam aquele comportamento. Sem soundtrack, o comportamento aprovado de continuação permanece igual.
