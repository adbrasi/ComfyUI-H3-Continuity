# Resultados — pesquisa de memória 01

RTX5090; FL2VA + Turbo, 8 steps, 736×416. Todos os testes da tabela entregam 102 frames novos. Inspeção por folhas de contato; as aprovações explícitas do usuário estão registradas no CHECKPOINT.md.

| Método | Seed89123 (s) | Seed89124 (s) | Aparência na reentrada, ambas as seeds |
|---|---:|---:|---|
| tail22 | 31.61 | 21.88 | Preto/branco, pelo comprido; falhou |
| tail90 | 39.74 | 39.49 | Azul, asas brancas, acessório amarelo; aproximou a fonte |
| encoder_image | 23.39 | 22.86 | Azul, asas brancas, acessório amarelo; aproximou a fonte |
| encoder_latent_image | 23.77 | 24.06 | Azul, asas brancas, acessório amarelo; aproximou a fonte |
| encoder_latent_small | 23.03 | 22.55 | Azul, asas brancas, acessório amarelo; aproximou a fonte |
| guide90 | 66.86 | 65.50 | Azul, asas brancas, acessório amarelo; aproximou a fonte |
| anchors90 | 61.20 | 61.32 | Azul, asas brancas, acessório amarelo; aproximou a fonte |
| pinned_pixels90 | 42.98 | 42.04 | Azul, asas brancas, acessório amarelo; aproximou a fonte |

Tempos de execução completos observados, incluindo preparação/decodificação/gravação, com caches e carregamentos do ComfyUI. O primeiro tail22 teve custo de inicialização maior. Não houve benchmark aleatorizado nem medição de pico de VRAM. O custo menor de pinned90 é coerente com não duplicar o vídeo de contexto em um bloco extra de guides, mas os valores não devem ser extrapolados para outras GPUs ou durações.

O Add Guide corretamente preparado **também recupera o cachorro**. Este teste não sustenta superioridade exclusiva do pack em memória ou qualidade visual. A máscara conserva o latent conhecido; o transporte direto evita recodificação; as referências esparsas recuperam aparência sem exigir 90 frames no target. O sampler e o encoder continuam nativos.

O encoder sozinho recuperou aparência em cerca de 23 s, comparado com cerca de 40 s do prefixo90. Isso não significa que todas as suas emendas sejam melhores. Há mudanças de enquadramento em alguns primeiros frames novos. Todas as variantes ainda precisam enfrentar prompts diferentes e câmera mais exigente.

Nos dois controles com referência trocada (seed 89123), o cachorro ganhou pernas mais altas, asas escuras e acessório roxo. O prefixo original não foi trocado. Os resultados estão nos vídeos `memory_wrong_*`.

O render `long_context_112_requested` completou em 48,962 s, usando 107 frames e gerando 102 novos, totalizando 226 na montagem. O erro máximo do prefixo em relação ao latent original foi 4,77e-7. A continuação conserva a aparência da mulher/parede; a marcha perde intensidade no fim, portanto contexto maior não garante atendimento perfeito ao prompt.

`memory_latent_metrics.json` compara o prefixo de cada resultado ao **latent bruto original**. No modo pinned_pixels90 esse erro inclui a perda da recodificação VAE, e não mede alteração causada pelo sampler; não se deve interpretá-lo como falha da máscara. Nos modos de transporte direto com máscara, o erro ficou em 4,77e-7. `memory_visual_metrics.json` contém métricas de diferença de pixels, não notas de qualidade.

46 testes de código passaram. As evidências GPU novas cobrem contextos 90 e107. Casos 124/243/362 dos testes unitários usam tensores sintéticos e não validam qualidade nesses comprimentos.
