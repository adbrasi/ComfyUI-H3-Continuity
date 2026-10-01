# Comparação de métodos de extensão — duelo

Vídeo: `/workspace/comfy/ComfyUI/output/h3_method_validation/comparacao_metodos_2x2.mp4`.

Quatro painéis sincronizados, 430 frames a 24 fps (17,92 segundos), 1920×1160. Sem áudio no mosaico; os vídeos individuais conservam o áudio. Uma versão por método, sem duplicar feather ligado/desligado.

| Posição | Método | Feather | Estratégia |
|---|---|---|---|
| Superior esquerda | pinned_av | 17 frames, força 1 | Fixa contexto anterior de vídeo e áudio, liberando a borda de vídeo. Reutiliza exatamente o vídeo aprovado pelo usuário. |
| Superior direita | pinned_prefix | 17 frames, força 1 | Fixa prefixo de vídeo; áudio segue o caminho sem máscara fixa do pinned_av. |
| Inferior esquerda | anchors | Não implementado | Usa contexto como guias latentes; parâmetro feather não participa desse caminho. |
| Inferior direita | bridge + âncora nova | 17 frames, força 1 | Gera intervalo com contexto nas duas pontas, conectando a origem a um clipe independente novo. |

## Controles e limites

Todos partem do mesmo latent do duelo (124 frames, 736×416), mesmo prompt, mesma referência canônica extraída do primeiro frame, contexto de 39 frames e Euler/simple com oito passos. Os três modos encadeados usam seeds 35702, 35703 e 35704, com três extensões de 102 frames entregues. O bridge usa seed 35702 para o intervalo e seed 35704 para a âncora independente: 124 + 182 + 124 = 430 frames. Não possui o mesmo orçamento de sampling nem a mesma topologia dos três encadeamentos; seeds iguais tampouco garantem coreografia igual.

Feather significa liberar parcialmente a borda latente durante sampling e entregar a borda regenerada. O mosaico não faz crossfade. A exceção anchors está explícita porque a implementação não tem feather; não foi apresentado um parâmetro ignorado como se estivesse funcionando.

Todas as quatro novas execuções (pinned_prefix, anchors, âncora futura e bridge) terminaram com status `success`. Histórico, APIs e parâmetros estão arquivados nesta pasta. O controle pinned_av vem da validação anterior em `milestones/action_validation` e não precisou ser regenerado.

As folhas de contato mostram os mesmos duelistas e pátio, com diferenças de câmera, poses e iluminação. Anchors ficou mais claro neste exemplo. Não há evidência aqui de eliminação universal da degradação nem de continuidade física perfeita; são três extensões e uma ponte em uma cena.

Métricas de diferença RGB e fluxo óptico estão em `metrics/`. Descrevem transições e não medem identidade, anatomia ou qualidade estética. Por isso não elegemos vencedor por menor erro entre frames.

O retake com mudança marcante fica para o próximo teste solicitado pelo usuário; não faz parte desta comparação.
