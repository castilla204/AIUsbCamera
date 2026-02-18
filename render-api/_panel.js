
const KEY = localStorage.getItem('API_KEY_CLIENTE') || '';
const _LET = ['A','B','C','D','X'];
const _STATE = {};   // jid → {dirty:bool}

// ── Polling resiliente con AbortController + backoff exponencial ───────────
let _pollFails = 0;
let _pollInflight = false;

async function poll() {
    if(_pollInflight) return;   // si el anterior aún corre, no acumular
    _pollInflight = true;
    const ctrl = new AbortController();
    const timeoutId = setTimeout(() => ctrl.abort(), 8000);  // tope 8s
    try {
        // limit=100 (default era 25 → causaba que los jobs viejos cayeran del
        // listado en cuanto entraban nuevos. Cuando el usuario tenía abierta una
        // fila "Ver/Editar" de un job histórico y un nuevo job lo empujaba fuera
        // del top 25, document.getElementById('corrrow-jid') devolvía null y el
        // restore de display:'' fallaba silenciosamente → la fila "se cerraba"
        // visualmente sin avisar).
        const r = await fetch('/api/jobs?key=' + KEY + '&limit=100', {signal: ctrl.signal});
        clearTimeout(timeoutId);
        if(!r.ok) throw new Error('HTTP ' + r.status);
        applyJobs(await r.json());
        if(_pollFails > 0) {
            _pollFails = 0;
            setRelayStatus('ok');
        }
    } catch(e) {
        clearTimeout(timeoutId);
        _pollFails++;
        if(_pollFails >= 2) setRelayStatus('down');
    } finally {
        _pollInflight = false;
    }
}

function setRelayStatus(state) {
    let el = document.getElementById('relay-status');
    if(!el) {
        el = document.createElement('div');
        el.id = 'relay-status';
        el.style.cssText = 'position:fixed;top:10px;right:10px;padding:6px 12px;'
                         + 'border-radius:6px;font-size:11px;font-weight:700;'
                         + 'z-index:9998;pointer-events:none';
        document.body.appendChild(el);
    }
    if(state === 'down') {
        el.style.background = '#dc3545'; el.style.color = '#fff';
        el.textContent = '⚠ Sin conexión al relay';
    } else {
        el.style.background = ''; el.style.color = '';
        el.textContent = '';
    }
}

// A10: helper para escapar valores que se interpolan dentro de `onclick="fn('${x}')"`.
// Si `x` contiene una comilla simple, sin escape rompe el atributo y permite
// inyectar HTML/JS. Aplica también a job_id (UUID hex normalmente seguro, pero
// si la persistencia se corrompe podría llegar cualquier cosa).
function jsAttr(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function applyJobs(jobs) {
    const area = document.getElementById('cards-area');
    const active = jobs.filter(j => j.status !== 'done' && j.status !== 'error');
    const hist   = jobs.filter(j => j.status === 'done'  || j.status === 'error');

    if(active.length > 0) {
        const emp = area.querySelector('.empty'); if(emp) emp.remove();
    } else if(!area.querySelector('.card')) {
        if(!area.querySelector('.empty'))
            area.innerHTML = '<div class="empty">Sin peticiones activas</div>';
    }

    // Stats
    const nRev  = jobs.filter(j=>j.status==='awaiting_review').length;
    const nPend = jobs.filter(j=>j.status==='pending').length;
    const nDone = jobs.filter(j=>j.status==='done').length;
    const nErr  = jobs.filter(j=>j.status==='error').length;
    document.getElementById('stat-rev').textContent = '✏️ ' + nRev + ' revisión';
    document.getElementById('stat-pen').textContent = '⏳ ' + nPend + ' procesando';
    document.getElementById('stat-ok').textContent  = '✅ ' + nDone + ' hechos';
    document.getElementById('stat-err').textContent = '❌ ' + nErr + ' errores';

    // Polling adaptativo: jobs activos (pending/awaiting_review) reducen el
    // intervalo a 1.5s para feedback rápido; sin nada activo subimos a 5s
    // para no martillear al servidor con polls que devuelven lo mismo.
    _activeJobsCount = nRev + nPend;

    // Active cards: crear las nuevas, actualizar las existentes
    const seen = new Set();
    // Aislamiento por job: si UN job malformado tira un error en buildCard/
    // updateCard, los demás siguen renderizándose. Antes un job con campos raros
    // (responses=null, _providers=string, fecha inválida) tumbaba el poll entero
    // y la lista quedaba congelada hasta el siguiente refresh manual.
    for(const job of active) {
        seen.add(job.id);
        try {
            const el = document.getElementById('card-' + job.id);
            if(!el) {
                area.insertAdjacentHTML('afterbegin', buildCard(job));
                ticks();
            } else {
                updateCard(el, job);
            }
        } catch(jobErr) {
            console.error('[panel] Error rendering active job ' + (job && job.id),
                          jobErr);
        }
    }
    // Retirar las que ya no están activas
    area.querySelectorAll('.card[data-jid]').forEach(el => {
        if(!seen.has(el.dataset.jid)) {
            el.style.transition = 'opacity .4s'; el.style.opacity = '0';
            setTimeout(() => el.remove(), 420);
        }
    });

    // Historial (con botón Corregir + ver respuestas individuales + vídeo)
    if(hist.length > 0) {
        const tb = document.getElementById('hist-body');

        // Preservar estado entre polls (cada 1.5s applyJobs re-render el tbody
        // entero; sin esto las filas "Ver/Editar" abiertas se cierran solas y
        // los inputs que el usuario esté rellenando se borran).
        const expandedIds = new Set();
        tb.querySelectorAll('tr[id^="corrrow-"]').forEach(tr => {
            if(tr.style.display !== 'none') expandedIds.add(tr.id.substring(8));
        });

        // A11: ANTES si había UNA fila expandida, se congelaba el historial
        // entero — nuevos jobs no aparecían hasta cerrar la fila. Ahora hacemos
        // re-render selectivo: actualizamos las filas que NO están expandidas
        // y preservamos las expandidas con su DOM intacto (sin tocar el <video>
        // ni el contenido cargado vía /api/partial).
        //
        // Estrategia:
        //   1) Si NO hay filas expandidas → fast path: re-render todo el tbody
        //      (igual que antes, conserva el comportamiento)
        //   2) Si HAY filas expandidas → re-render por fila:
        //        - para cada job del top-30, busca su <tr id="hist-{jid}"> existente
        //        - si jid está en expandedIds → no toques la fila ni su corrrow
        //        - si no → reemplaza solo esa fila + su corrrow vacía
        //        - los jobs nuevos se insertan al principio
        //        - los jobs viejos (fuera del top-30) se quitan SI no están expandidos
        const savedInputs = {};   // id → value (inputs/textareas dentro de filas expandidas)
        const savedFocusId = (document.activeElement && document.activeElement.id) || null;
        const savedSelStart = (document.activeElement && 'selectionStart' in document.activeElement)
                              ? document.activeElement.selectionStart : null;
        const savedSelEnd   = (document.activeElement && 'selectionEnd' in document.activeElement)
                              ? document.activeElement.selectionEnd : null;
        expandedIds.forEach(jid => {
            const row = document.getElementById('corrrow-' + jid);
            if(!row) return;
            row.querySelectorAll('input, textarea, select').forEach(el => {
                if(el.id) savedInputs[el.id] = el.value;
            });
        });

        // Helper: construye el par <tr hist> + <tr corrrow vacío> para UN job.
        // try/catch interno → un job malformado no tira el render entero.
        function _renderHistRowPair(j) {
            try {
                const hora = new Date((j.created || 0) * 1000).toLocaleTimeString('es');
                const bg   = j.status === 'done' ? '#198754' : '#dc3545';
                const ans  = (j.answer || '-').toUpperCase();
                const m    = (j.merged_answer || '').toUpperCase();
                const auto = j.auto_approved ? ' 🤖' : '';
                const canCorrect = (j.status === 'done' || j.status === 'error');
                const jidEsc = jsAttr(j.id);
                const ansEsc = jsAttr(ans);
                const correctBtn = canCorrect
                    ? `<button class="hist-corr" onclick="toggleCorr('${jidEsc}','${ansEsc}')">✏️ Ver/Editar</button>`
                    : '';
                const tStats = j.tavily_stats || null;
                const hasTavQ = tStats && Array.isArray(tStats.queries) && tStats.queries.length > 0;
                if(hasTavQ) {
                    window._TAVILY_STATS = window._TAVILY_STATS || {};
                    window._TAVILY_STATS[j.id] = tStats;
                }
                const tavBtn = hasTavQ
                    ? `<button class="hist-corr" onclick="showTavilyQueries('${jidEsc}')"
                               style="background:#0ea5e9;border:none;color:#fff;margin-right:4px"
                               title="Ver búsquedas Tavily inyectadas al razonador">🔎 ${tStats.queries.length}</button>`
                    : '';
                const statusUpper = jsAttr(String(j.status || 'unknown').toUpperCase());
                return `<tr id="hist-${jidEsc}">
                    <td class="gray">${jsAttr(hora)}</td>
                    <td><span style="padding:2px 8px;border-radius:12px;background:${bg};
                        font-size:10px;font-weight:700;color:#fff">${statusUpper}${auto}</span></td>
                    <td class="mono">${jsAttr(ans)}</td>
                    <td class="mono gray">${m && m!==ans ? jsAttr(m) : ''}</td>
                    <td style="text-align:right">${tavBtn}${correctBtn}</td>
                </tr>
                <tr id="corrrow-${jidEsc}" style="display:none" data-loaded="0">
                    <td colspan="5" style="background:#0d0d0d;padding:14px">
                        <div style="color:#888;text-align:center;padding:14px;font-size:12px">
                            ⏳ Pulsa "✏️ Ver/Editar" para cargar el expediente completo
                        </div>
                    </td>
                </tr>`;
            } catch(histErr) {
                console.error('[panel] Error rendering history job ' + (j && j.id),
                              histErr);
                return `<tr><td colspan="5" style="color:#dc3545;font-size:11px">
                    ⚠ Fila inválida (job ${j && j.id ? j.id.slice(0,8) : '?'})
                    </td></tr>`;
            }
        }

        const top30 = hist.slice(0, 30);
        if(expandedIds.size === 0) {
            // FAST PATH: ninguna fila expandida → re-render limpio del tbody.
            tb.innerHTML = top30.map(_renderHistRowPair).join('');
        } else {
            // A11: SELECTIVE PATH. Hay filas expandidas → no destruir el DOM
            // existente. Por cada job del top-30:
            //   - si su hist-row ya existe → actualiza SOLO los <td> de la fila
            //     principal (no toca el corrrow expandido)
            //   - si no existe → la creamos (inserta al principio)
            // Y quitamos las filas históricas que ya no estén en el top-30
            // (salvo las que estén expandidas — esas se quedan hasta cerrarlas).
            const seenIds = new Set();
            let prevRow = null;   // para inserción ordenada
            for(const j of top30) {
                seenIds.add(j.id);
                const existing = document.getElementById('hist-' + j.id);
                if(existing) {
                    // Actualiza solo la fila hist; el corrrow ya tiene su contenido.
                    // Reemplazamos el <tr id="hist-..."> con el nuevo, manteniendo
                    // el corrrow intacto.
                    const tmp = document.createElement('tbody');
                    tmp.innerHTML = _renderHistRowPair(j);
                    const newHistRow = tmp.querySelector('tr[id^="hist-"]');
                    if(newHistRow) {
                        existing.replaceWith(newHistRow);
                        // El corrrow queda donde estaba (después del histRow nuevo
                        // gracias a su posición previa).
                    }
                    prevRow = document.getElementById('corrrow-' + j.id) || existing;
                } else {
                    // Insertar nuevo pair al principio (o tras prevRow si hay)
                    const tmp = document.createElement('tbody');
                    tmp.innerHTML = _renderHistRowPair(j);
                    const newRows = Array.from(tmp.children);
                    if(prevRow && prevRow.parentNode === tb) {
                        for(const nr of newRows) prevRow.after(nr);
                    } else {
                        for(let i = newRows.length - 1; i >= 0; i--) {
                            tb.insertAdjacentElement('afterbegin', newRows[i]);
                        }
                    }
                    prevRow = newRows[newRows.length - 1];
                }
            }
            // Quitar filas viejas que ya no están en top30, EXCEPTO expandidas
            Array.from(tb.querySelectorAll('tr[id^="hist-"]')).forEach(tr => {
                const jid = tr.id.substring(5);
                if(!seenIds.has(jid) && !expandedIds.has(jid)) {
                    const corrrow = document.getElementById('corrrow-' + jid);
                    tr.remove();
                    if(corrrow) corrrow.remove();
                }
            });
        }

        // Restaurar visibilidad de las filas que estaban abiertas + valores que
        // el usuario haya tipeado dentro. Si el job ya no existe (cayó del top
        // 30), se ignora silenciosamente.
        expandedIds.forEach(jid => {
            const row = document.getElementById('corrrow-' + jid);
            if(row) row.style.display = '';
        });
        Object.keys(savedInputs).forEach(id => {
            const el = document.getElementById(id);
            if(el) el.value = savedInputs[id];
        });
        // Restaurar foco (con caret/selección si era input/textarea) para que
        // el usuario no pierda el cursor en medio de una corrección.
        if(savedFocusId) {
            const el = document.getElementById(savedFocusId);
            if(el) {
                el.focus();
                if(savedSelStart != null && 'setSelectionRange' in el) {
                    try { el.setSelectionRange(savedSelStart, savedSelEnd ?? savedSelStart); }
                    catch(_) {}
                }
            }
        }
    }
}

// ── Detalle de un job histórico ──────────────────────────────────────────────
// EXPEDIENTE COMPLETO: media + OCRs (si vídeo) + respuestas IA (con raw, modelo,
// error, edición de letras) + matriz de votos de la fusión + editor de
// corrección retroactiva. Todos los datos vienen ya en el job (responses,
// ocr_results, ocr_fused_text, etc.) — solo los renderizamos.
function buildHistoryDetail(j, ans) {
    const jid        = j.id;
    const responses  = j.responses || {};
    const ocrResults = j.ocr_results || {};
    const merged     = (j.merged_answer || '').toUpperCase();
    const expected   = j.expected_questions || Math.max(merged.length, ans.length, 1);
    // A10: añadimos escape de comilla simple (&#39;) — antes solo se escapaban
    // &<>" — así un dato corrupto con `'` que llegue a un `onclick="fn('${x}')"`
    // no rompe el atributo ni permite XSS. Coste cero, defensa universal.
    const escape     = s => String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')
                                          .replace(/>/g,'&gt;').replace(/"/g,'&quot;')
                                          .replace(/'/g,'&#39;');

    // 1) Vídeo / imagen del job (si todavía está en RAM)
    let mediaHtml = '';
    if(j.has_video && !j.video_pruned) {
        mediaHtml = `<div style="margin-bottom:10px">
            <video src="/api/video/${jid}?key=${KEY}" controls muted playsinline
                   style="width:100%;max-width:480px;max-height:240px;border-radius:6px;background:#000"
                   onerror="this.parentNode.innerHTML='<div style=color:#888;font-size:11px>⚠ Vídeo expirado (>30min)</div>'">
            </video>
        </div>`;
    } else if(j.has_video && j.video_pruned) {
        mediaHtml = `<div style="color:#888;font-size:11px;margin-bottom:8px">
            ⏳ Vídeo expirado (purgado tras 30min para liberar RAM)
        </div>`;
    } else if(j.has_img && !j.img_pruned) {
        mediaHtml = `<div style="margin-bottom:10px">
            <img src="/api/image/${jid}?key=${KEY}" style="max-width:100%;max-height:240px;border-radius:6px">
        </div>`;
    }

    // 2) OCRs por IA + fusión OCR (sólo en jobs de vídeo)
    let ocrHtml = '';
    if(j.has_video) {
        ocrHtml = '<div style="margin-bottom:12px">'
                + '<div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px;'
                + 'padding:4px 0">📝 OCRs del vídeo (fase 1)</div>';
        const ocrProvs = ['qwen_ocr','gemini_ocr','kimi_ocr','mimo_ocr','anthropic_ocr','openai_ocr','nvidia_ocr'];
        for(const prov of ocrProvs) {
            const r = ocrResults[prov];
            const label = PROVIDER_LABELS[prov] || prov;
            if(!r) {
                ocrHtml += `<div style="display:flex;gap:8px;padding:3px 0;font-family:monospace;font-size:12px;color:#555">
                    <span style="min-width:120px">${label}</span><span>— sin datos</span></div>`;
            } else if(r.ok) {
                // Mismo patrón defensivo que buildOcrRow: si backend viejo no
                // envía ni chars ni text en el slim, mostramos "OK" en lugar
                // de "0 chars" para no inducir a error.
                let chars, charsLabel;
                if(r.chars != null) {
                    chars = r.chars;
                    charsLabel = `${chars} chars`;
                } else if(r.text != null) {
                    chars = r.text.length;
                    charsLabel = `${chars} chars`;
                } else {
                    chars = null;
                    charsLabel = 'OK';
                }
                if(r.text) {
                    window._OCR_TEXTS = window._OCR_TEXTS || {};
                    window._OCR_TEXTS[`${jid}|${prov}`] = r.text;
                }
                const previewSrc = r.text || r.preview || '';
                const preview = escape(previewSrc.slice(0, 50));
                const ms = r.ms ? `<small style="color:#666">${ms_fmt(r.ms)}</small>` : '';
                ocrHtml += `<div style="display:flex;gap:8px;align-items:center;padding:3px 0;font-family:monospace;font-size:12px">
                    <span style="min-width:120px;color:#9cf">${label} ${ms}</span>
                    <span style="color:#5d8;min-width:80px">✓ ${charsLabel}</span>
                    <span style="color:#888;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${preview}${(chars != null && chars > 50) ? '…' : ''}</span>
                    <button onclick="showOcrText('${jid}','${prov}','${label}')"
                            style="background:#0d6efd;color:#fff;border:none;border-radius:4px;
                                   padding:3px 8px;font-size:10px;cursor:pointer" title="Ver transcripción completa">📜</button>
                </div>`;
            } else {
                const err = escape((r.error || 'error').slice(0, 100));
                const ms = r.ms ? `<small style="color:#666">${ms_fmt(r.ms)}</small>` : '';
                ocrHtml += `<div style="display:flex;gap:8px;padding:3px 0;font-family:monospace;font-size:12px">
                    <span style="min-width:120px;color:#9cf">${label} ${ms}</span>
                    <span style="color:#dc3545" title="${err}">❌ ${err}</span></div>`;
            }
        }
        // Fila de la fusión OCR (texto canónico que se envió a los analyzers)
        if(j.ocr_fused_text) {
            window._OCR_TEXTS = window._OCR_TEXTS || {};
            // Sólo cacheamos la versión COMPLETA. En el listado slim el backend
            // trunca ocr_fused_text a 200 chars y marca _ocr_fused_truncated=true;
            // si cacheamos eso, el popup mostraría 200 chars en lugar de los 4000+
            // reales. Mejor dejar vacío y que showOcrText haga lazy-fetch desde
            // /api/partial cuando el usuario abra el modal.
            if(!j._ocr_fused_truncated) {
                window._OCR_TEXTS[`${jid}|fusion`] = j.ocr_fused_text;
            }
            const used = j.ocr_fusion_used === true;
            const stats = j.ocr_fusion_stats || {};
            const k = stats.clusters || 0;
            const color = used ? '#5d8' : '#e0c060';
            const tag   = used ? '✓ usada' : '⚠ no usada (fallback)';
            // chars REAL desde ocr_fused_text_full_chars; .length daría el truncado.
            const fusedChars = (j.ocr_fused_text_full_chars != null)
                                 ? j.ocr_fused_text_full_chars
                                 : j.ocr_fused_text.length;
            ocrHtml += `<div style="display:flex;gap:8px;align-items:center;padding:5px 0 3px;
                                     border-top:1px dashed #2a2a2a;margin-top:4px;font-family:monospace;font-size:12px">
                <span style="min-width:120px;color:${color};font-weight:700">🧬 Fusión OCR</span>
                <span style="color:${color};flex:1">${tag} · ${k} preguntas · ${fusedChars} chars</span>
                <button onclick="showOcrText('${jid}','fusion','Fusión OCR')"
                        style="background:${used?'#198754':'#d4a017'};color:#fff;border:none;border-radius:4px;
                               padding:3px 8px;font-size:10px;cursor:pointer">📜</button>
            </div>`;
        }
        // Fila Tavily (paso intermedio OCR → razonamiento). Mostramos qué se buscó,
        // si llegó a tiempo, cuántas fuentes salieron, y permitimos ver el XML
        // exacto que se inyectó al prompt del razonador.
        const tavStats = j.tavily_stats || {};
        const tavEnabled = (tavStats.enabled === true) || j.tavily_used === true;
        if(tavEnabled || tavStats.skipped) {
            const nQ        = tavStats.n_questions   || 0;
            const nSources  = tavStats.n_with_sources || 0;
            const elapsed   = tavStats.elapsed_ms    || 0;
            const skipped   = tavStats.skipped       || '';
            const blockLen  = (j.tavily_block_full_chars != null)
                                ? j.tavily_block_full_chars
                                : (j.tavily_block ? j.tavily_block.length : 0);
            const usedTav   = j.tavily_used === true;
            const tCol      = skipped ? '#888' : (usedTav ? '#0ea5e9' : '#e0c060');
            let tDesc;
            if(skipped) {
                tDesc = `⏭ omitido · motivo=${escape(skipped)}`;
            } else if(usedTav) {
                tDesc = `✓ inyectado · ${nSources}/${nQ} preguntas con fuentes · `
                      + `${blockLen} chars · ${ms_fmt(elapsed)}`;
            } else {
                tDesc = `⚠ sin resultados utilizables · ${nQ} preguntas · ${ms_fmt(elapsed)}`;
            }
            // Cachea el bloque para que el modal no haga lazy-fetch si ya lo tenemos.
            if(j.tavily_block && !j._tavily_block_truncated) {
                window._TAVILY_BLOCKS = window._TAVILY_BLOCKS || {};
                window._TAVILY_BLOCKS[jid] = j.tavily_block;
            }
            window._TAVILY_STATS = window._TAVILY_STATS || {};
            if(Array.isArray(tavStats.queries) && tavStats.queries.length) {
                window._TAVILY_STATS[jid] = tavStats;
            }
            const btns = [];
            if(Array.isArray(tavStats.queries) && tavStats.queries.length) {
                btns.push(`<button onclick="showTavilyQueries('${jid}')"
                                    style="background:#0ea5e9;color:#fff;border:none;border-radius:4px;
                                           padding:3px 8px;font-size:10px;cursor:pointer"
                                    title="Ver qué buscó Tavily por cada pregunta">🔎 ${tavStats.queries.length}</button>`);
            }
            if(blockLen > 0) {
                btns.push(`<button onclick="showTavilyBlock('${jid}')"
                                    style="background:#0d6efd;color:#fff;border:none;border-radius:4px;
                                           padding:3px 8px;font-size:10px;cursor:pointer"
                                    title="Ver el XML <internet_context> EXACTO inyectado al prompt del razonador">📜</button>`);
            }
            ocrHtml += `<div style="display:flex;gap:8px;align-items:center;padding:5px 0 3px;
                                     border-top:1px dashed #2a2a2a;margin-top:4px;font-family:monospace;font-size:12px">
                <span style="min-width:120px;color:${tCol};font-weight:700">🌐 Tavily</span>
                <span style="color:${tCol};flex:1">${tDesc}</span>
                ${btns.join('')}
            </div>`;
        }
        ocrHtml += '</div>';
    }

    // 3) Respuestas individuales por IA (editables · con raw / modelo / error)
    let respsHtml = '<div style="margin-bottom:12px">'
                  + '<div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px;'
                  + 'padding:4px 0">🧠 Respuestas por IA '
                  + '<small style="text-transform:none;letter-spacing:0;color:#666">(💾 guarda letras editadas · 📜 ve la respuesta completa)</small>'
                  + '</div>';
    const provs = (j._providers && j._providers.length) ? j._providers
                 : Object.keys(responses);
    if(provs.length === 0) {
        respsHtml += '<div style="color:#666;font-size:11px">(sin datos de IAs)</div>';
    } else {
        for(const prov of provs) {
            const r     = responses[prov] || {};
            const label = (typeof PROVIDER_LABELS!=='undefined' && PROVIDER_LABELS[prov])
                         || prov.toUpperCase();
            const a     = (r.answer || '').toUpperCase();
            const okIco = r.ok ? '✓' : (r.status==='no_key' ? '🔑' : '❌');
            const ms    = r.ms ? `<small style="color:#666">${ms_fmt(r.ms)}</small>` : '';
            const edited = r.edited ? '<small style="color:#ffc107"> · ✎ editado</small>' : '';
            const modelTag = r.model
                ? `<small style="color:#557;font-size:10px" title="modelo usado">[${escape(r.model)}]</small>`
                : '';

            // Botón "📜" para abrir el raw completo de esta IA en modal.
            // Cacheamos en window._RAW_RESPONSES para no meter 5000+ chars al DOM por
            // cada IA × cada job × top-30 del historial.
            let rawBtn = '';
            if(r.raw && r.raw.length > 0) {
                window._RAW_RESPONSES = window._RAW_RESPONSES || {};
                window._RAW_RESPONSES[`${jid}|${prov}`] = r.raw;
                rawBtn = `<button onclick="showRawResponse('${jid}','${prov}','${label}')"
                                  style="background:#6c757d;color:#fff;border:none;border-radius:4px;
                                         padding:4px 8px;font-size:10px;cursor:pointer"
                                  title="Ver razonamiento/respuesta completa de ${label} (${r.raw.length} chars)">📜</button>`;
            }

            // Mensaje de error inline si la IA falló (status_no_key no es error real)
            const errInline = (!r.ok && r.status !== 'no_key' && r.error)
                ? `<div style="color:#dc3545;font-size:11px;padding:2px 0 0 98px;font-family:monospace"
                        title="${escape(r.error)}">⚠ ${escape((r.error||'').slice(0,180))}</div>`
                : '';
            // Mensaje no_key (gris, no es error)
            const noKeyInline = (r.status === 'no_key')
                ? `<div style="color:#888;font-size:11px;padding:2px 0 0 98px;font-style:italic">
                       🔑 sin key configurada — no participa en la fusión</div>`
                : '';

            respsHtml += `<div>
                <div style="display:flex;gap:8px;align-items:center;padding:4px 0;
                            font-family:monospace;font-size:13px;flex-wrap:wrap">
                    <span style="min-width:90px;color:#9cf">${label} ${okIco}</span>
                    <input id="resp-${jid}-${prov}" type="text" maxlength="50"
                           value="${a}"
                           style="flex:1;min-width:80px;background:#1a1a1a;color:#fff;
                                  border:1px solid #333;border-radius:4px;padding:4px 8px;
                                  font-family:monospace;font-weight:700;letter-spacing:2px;text-transform:uppercase"
                           oninput="this.value=this.value.toUpperCase().replace(/[^ABCDX]/g,'')">
                    <button onclick="saveProviderResponse('${jid}','${prov}')"
                            style="background:#0d6efd;color:#fff;border:none;border-radius:4px;
                                   padding:4px 10px;font-size:11px;font-weight:700;cursor:pointer"
                            title="Guardar letras editadas (recalcula la fusión)">💾</button>
                    ${rawBtn}
                    ${ms}${edited}${modelTag}
                </div>
                ${errInline}${noKeyInline}
            </div>`;
        }
        // Fila fusión + botón para mostrar matriz de votos
        respsHtml += `<div style="display:flex;gap:8px;align-items:center;padding:6px 0;
                                   font-family:monospace;font-size:13px;border-top:1px dashed #2a2a2a;margin-top:4px">
            <span style="min-width:90px;color:#5d8;font-weight:700">🔀 Fusión</span>
            <span id="merged-${jid}" style="flex:1;font-family:monospace;letter-spacing:2px;font-weight:700;color:#fff">${merged || '—'}</span>
            <button onclick="toggleFusionMatrix('${jid}')"
                    style="background:#5d8;color:#000;border:none;border-radius:4px;
                           padding:3px 8px;font-size:10px;font-weight:700;cursor:pointer"
                    title="Ver / ocultar la matriz de votos pregunta-a-pregunta">📊 Votos</button>
            <small style="color:#666;font-size:10px">(auto-recalcula)</small>
        </div>
        <div id="fusion-matrix-${jid}" style="display:none;margin-top:6px">${buildFusionMatrix(j)}</div>`;
    }
    respsHtml += '</div>';

    // 4) Editor de corrección retroactiva (vibra al móvil)
    const corrHtml = `<div style="border-top:1px solid #2a2a2a;padding-top:10px">
        <div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px;padding-bottom:6px">
            📤 Enviar corrección al móvil (vibrará la nueva respuesta)
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
            <input id="corra-${jid}" type="text" maxlength="30" value="${ans}"
                   style="flex:1;min-width:160px;font-family:monospace;font-weight:700;
                          font-size:18px;letter-spacing:3px;text-transform:uppercase;
                          padding:8px;background:#1a1a1a;color:#fff;border:2px solid #fd7e14;border-radius:6px"
                   oninput="this.value=this.value.toUpperCase().replace(/[^ABCDX]/g,'')">
            <input id="corrp-${jid}" type="number" min="1" max="99" placeholder="Pág"
                   style="width:70px;padding:8px;background:#1a1a1a;color:#fff;
                          border:2px solid #2a2a2a;border-radius:6px;text-align:center">
            <button onclick="sendCorrection('${jid}')"
                    style="background:#fd7e14;color:#fff;border:none;border-radius:6px;
                           padding:9px 14px;font-weight:700;cursor:pointer">📤 Enviar corrección</button>
            <span id="corrs-${jid}" style="font-size:11px;color:#888"></span>
        </div>
    </div>`;

    return mediaHtml + ocrHtml + respsHtml + corrHtml;
}

// Toggle de visibilidad de la matriz de votos en el detalle del job.
function toggleFusionMatrix(jid) {
    const el = document.getElementById(`fusion-matrix-${jid}`);
    if(!el) return;
    el.style.display = (el.style.display === 'none' || !el.style.display) ? 'block' : 'none';
}

// Matriz de votos pregunta-a-pregunta: filas = IAs, columnas = preguntas.
// Resalta en verde las letras que coinciden con la fusión, en rojo las que disienten,
// en gris las X / sin opinión. La fila inferior es la fusión final.
function buildFusionMatrix(j) {
    const merged    = (j.merged_answer || '').toUpperCase();
    const responses = j.responses || {};
    const provs     = (j._providers && j._providers.length)
                     ? j._providers : Object.keys(responses);
    if(provs.length === 0) {
        return '<div style="color:#666;font-size:11px;padding:6px">Sin datos de votos.</div>';
    }
    // Longitud objetivo: la mayor entre fusión, expected, y la respuesta más larga.
    let maxLen = j.expected_questions || merged.length || 0;
    for(const prov of provs) {
        const a = (responses[prov] && responses[prov].answer) || '';
        if(a.length > maxLen) maxLen = a.length;
    }
    if(maxLen === 0) {
        return '<div style="color:#666;font-size:11px;padding:6px">Aún sin respuestas para votar.</div>';
    }

    let html = '<div style="overflow-x:auto;border:1px solid #2a2a2a;border-radius:6px;'
             + 'padding:8px;background:#0d0d0d">'
             + '<div style="font-size:10px;color:#888;padding-bottom:6px">'
             + '📊 Matriz de votos · una columna por pregunta. '
             + '<span style="color:#5d8">verde</span>=coincide con fusión · '
             + '<span style="color:#dc3545">rojo</span>=disiente · '
             + '<span style="color:#888">·</span>=sin opinión (X o no respondió).'
             + '</div>'
             + '<table style="border-collapse:collapse;font-family:monospace;font-size:11px">'
             + '<thead><tr><th style="text-align:left;padding:3px 8px;color:#888">IA</th>';
    for(let i = 0; i < maxLen; i++) {
        html += `<th style="padding:3px 5px;color:#888;min-width:18px;text-align:center">${i+1}</th>`;
    }
    html += '<th style="padding:3px 8px;color:#888;text-align:right">estado</th></tr></thead><tbody>';

    for(const prov of provs) {
        const r = responses[prov] || {};
        const label = (typeof PROVIDER_LABELS!=='undefined' && PROVIDER_LABELS[prov])
                     || prov.toUpperCase();
        const ans = (r.answer || '').toUpperCase();
        let estado = '—';
        if(r.status === 'no_key') estado = '🔑';
        else if(r.ok)            estado = '✓';
        else if(r.status === 'error') estado = '❌';
        html += `<tr><td style="padding:3px 8px;color:#9cf;white-space:nowrap">${label}</td>`;
        for(let i = 0; i < maxLen; i++) {
            const ch = (i < ans.length) ? ans[i] : '';
            const m  = (i < merged.length) ? merged[i] : '';
            let color = '#666';
            let bg    = 'transparent';
            let disp  = ch || '·';
            if(!ch || ch === 'X') { color = '#666'; disp = ch || '·'; }
            else if(!m)            { color = '#aaa'; }
            else if(ch === m)      { color = '#5d8'; bg = '#0d1a0d'; }
            else                   { color = '#dc3545'; bg = '#1a0d0d'; }
            html += `<td style="padding:3px 5px;text-align:center;color:${color};background:${bg};font-weight:700">${disp}</td>`;
        }
        html += `<td style="padding:3px 8px;text-align:right;color:#888">${estado}</td></tr>`;
    }
    // Fila de la fusión (resultado final)
    html += '<tr style="border-top:1px dashed #2a2a2a">'
          + '<td style="padding:4px 8px;color:#5d8;font-weight:700">🔀 FUSIÓN</td>';
    for(let i = 0; i < maxLen; i++) {
        const m = (i < merged.length) ? merged[i] : '·';
        html += `<td style="padding:4px 5px;text-align:center;color:#fff;font-weight:700;background:#1a3a1a">${m}</td>`;
    }
    html += '<td></td></tr></tbody></table></div>';
    return html;
}

// ── Guardar la respuesta editada de UN provider ──────────────────────────────
async function saveProviderResponse(jid, prov) {
    const el = document.getElementById(`resp-${jid}-${prov}`);
    if(!el) return;
    const ans = (el.value || '').toUpperCase().replace(/[^ABCDX]/g, '');
    if(!ans) { toast('⚠ Vacío — escribe letras A/B/C/D/X'); return; }
    try {
        const r = await fetch(`/api/response/${jid}/${prov}`, {
            method:  'PATCH',
            headers: {'Content-Type':'application/json','X-Api-Key':KEY},
            body:    JSON.stringify({answer: ans}),
        });
        let data = {}; try { data = await r.json(); } catch(_) {}
        if(!r.ok) throw new Error(data.detail || ('HTTP ' + r.status));
        // Actualizar la celda de fusión en sitio (sin recargar todo)
        const mEl = document.getElementById(`merged-${jid}`);
        if(mEl && data.merged_answer) mEl.textContent = data.merged_answer.toUpperCase();
        toast(`✓ ${prov}: ${data.old||'?'} → ${data.new}`);
    } catch(e) {
        toast('❌ ' + e.message);
    }
}

// ── Editor de correcciones en historial ──────────────────────────────────────
// Toggle de la fila expandida del historial. La primera vez que se abre carga
// el expediente completo del job via /api/partial/JID (que sí incluye raws de
// IA + transcripciones OCR completas, a diferencia del /api/jobs ligero).
// Hits sucesivos solo togglean visibilidad (cacheado en el DOM).
async function toggleCorr(jid, defaultAns) {
    const row = document.getElementById('corrrow-' + jid);
    if(!row) return;
    const opening = row.style.display === 'none' || !row.style.display;
    row.style.display = opening ? '' : 'none';
    if(!opening) return;            // cierre: no hace falta cargar nada
    if(row.dataset.loaded === '1') return;  // ya cargado en una apertura previa
    const td = row.querySelector('td');
    if(!td) return;
    td.innerHTML = '<div style="color:#888;text-align:center;padding:18px;font-size:12px">'
                 + '⏳ Cargando expediente del job...</div>';
    try {
        const r = await fetch('/api/partial/' + jid + '?key=' + encodeURIComponent(KEY),
                              { headers: { 'X-Api-Key': KEY } });
        if(!r.ok) {
            let detail = '';
            try { detail = (await r.json()).detail || ''; } catch(_) {}
            // 404 es un caso esperado: el job expiró (>1h) o el servidor reinició
            // y el JOBS dict en RAM se vació. Lanzamos un error con código para
            // que el catch lo distinga del resto y muestre un mensaje claro.
            const err = new Error('HTTP ' + r.status + (detail ? ' · ' + detail : ''));
            err.httpStatus = r.status;
            throw err;
        }
        const fullJob = await r.json();
        td.innerHTML = buildHistoryDetail(fullJob, defaultAns);
        row.dataset.loaded = '1';
    } catch(e) {
        // Render defensivo: si falla la carga, mostrar error con botón reintentar
        // que limpia el cache y vuelve a llamar a toggleCorr.
        const safeAns = String(defaultAns||'').replace(/'/g, '');
        // Si es 404 (job purgado / expirado), explicamos qué pasó en vez del HTTP
        // crudo. Reintentar no va a recuperar el job; sugerimos los datos básicos
        // que sí están en el listado del historial (answer, estado, IA fusión).
        if(e.httpStatus === 404) {
            td.innerHTML =
                '<div style="color:#e0c060;padding:14px;font-size:12px;line-height:1.5">'
              + '<div style="font-weight:700;margin-bottom:6px">📭 Expediente ya no disponible</div>'
              + '<div style="color:#aaa">Este job ya no está en memoria del servidor. Causas habituales:</div>'
              + '<ul style="color:#aaa;margin:6px 0 8px 18px;padding:0">'
              + '<li>Tiene más de 1h (TTL) — se purgó automáticamente.</li>'
              + '<li>El servicio se reinició (re-deploy de Render) y el dyno free perdió el disco.</li>'
              + '<li>Superó el cap de 150 jobs y se descartó por ser de los más antiguos.</li>'
              + '</ul>'
              + '<div style="color:#888;font-size:11px">Los datos básicos (respuesta, estado, IA fusión) siguen visibles en la fila del historial. '
              + 'El detalle completo (raws de IA, OCR, bloque Tavily) se perdió.</div>'
              + '</div>';
            return;
        }
        td.innerHTML =
            '<div style="color:#dc3545;padding:14px;font-size:12px">'
          + '⚠ No se pudo cargar el expediente: ' + (e.message || 'error desconocido') + '. '
          // FIX: usar &quot; HTML entities en lugar de \' — el JS exterior usa
          // comillas simples como delimitador, así que cada \' literal dentro
          // CERRABA el string JS y rompía el parseo de TODO el <script>. Sin
          // parsear, autoLoadConfig() ni loadConfig() corrían → panel sin keys.
          // Con &quot;, el navegador parsea HTML → JS recibe r.style.display="none"
          // (comillas dobles dentro de string ' es legal y no choca).
          + '<a onclick="(function(r){r.dataset.loaded=0;r.style.display=&quot;none&quot;;'
          +     'toggleCorr(&quot;' + jid + '&quot;,&quot;' + safeAns + '&quot;);})(this.closest(&quot;tr&quot;))" '
          +    'style="color:#4d9eff;cursor:pointer;text-decoration:underline">↻ Reintentar</a>'
          + '</div>';
    }
}

async function sendCorrection(jid) {
    const ansEl = document.getElementById('corra-' + jid);
    const pgEl  = document.getElementById('corrp-' + jid);
    const stEl  = document.getElementById('corrs-' + jid);
    const ans = (ansEl?.value || '').toUpperCase().replace(/[^ABCDX]/g, '');
    const page = pgEl?.value ? parseInt(pgEl.value) : null;
    if(!ans) { if(stEl) stEl.textContent = '⚠ Escribe la corrección'; return; }
    try {
        const r = await fetch('/correction/' + jid, {
            method:  'POST',
            headers: {'Content-Type': 'application/json', 'X-Api-Key': KEY},
            body:    JSON.stringify({answer: ans, page: page})
        });
        if(!r.ok) throw new Error((await r.json()).detail || r.status);
        const data = await r.json();
        if(stEl) stEl.textContent = '✅ Encolada · ' + data.answer + (page ? ' · pág ' + page : '');
        toast('📤 Corrección enviada');
        setTimeout(() => { const row = document.getElementById('corrrow-' + jid);
                           if(row) row.style.display = 'none'; }, 2000);
    } catch(e) {
        if(stEl) stEl.textContent = '❌ ' + e.message;
    }
}

// ── Construir tarjeta (primera vez) ──────────────────────────────────────────
function buildCard(job) {
    const jid      = job.id;
    const merged   = job.merged_answer || '';
    const expected = job.expected_questions || 0;
    const cols     = Math.max(expected, merged.length, 1);
    const hora     = new Date(job.created * 1000).toLocaleTimeString('es');
    const dl       = job.review_deadline || 0;
    const tout     = job.review_timeout_seconds || 90;
    _STATE[jid]    = {dirty: false};

    const imgHtml = job.has_img
        ? `<div class="img-wrap">
               <img src="/api/image/${jid}?key=${KEY}" class="foto"
                    onclick="this.classList.toggle('big')" alt="Foto">
           </div>`
        : '';

    // Video MP4 reproducible — sólo si el job lo trae (mode VIDEO). Se carga
    // desde /api/video. Permite ver EXACTAMENTE lo que se envió a OCR.
    const videoHtml = job.has_video
        ? `<div class="img-wrap" style="background:#000;border:1px solid #2a2a2a;border-radius:8px;margin:6px 0">
               <video src="/api/video/${jid}?key=${KEY}" controls muted playsinline class="card-video"
                      onerror="this.parentNode.innerHTML='<div style=padding:10px;color:#888>⚠ Video no disponible (probablemente perdido tras reinicio)</div>'">
               </video>
               <div style="font-size:10px;color:#888;padding:4px 6px;text-align:right">
                   📹 ${Math.round((job.video_size_b64||0)*3/4/1024)} KB · MP4
               </div>
           </div>`
        : '';

    return `
    <div class="card card-pend" id="card-${jid}" data-jid="${jid}"
         data-status="${job.status}" data-merged="${merged}" data-cols="${cols}">
        <div class="card-head">
            <span class="badge b-pend" id="badge-${jid}">⏳ PROCESANDO</span>
            <span class="hora">${hora}</span>
            <span class="cd" id="cd-${jid}" data-deadline="${dl}" data-total="${tout}">⏱</span>
        </div>
        <div class="phase-bar" id="phase-${jid}" style="font-size:11px;color:#9ad;padding:4px 8px;background:#0d1620;border-radius:5px;margin:4px 0">
            ${renderPhase(job)}
        </div>
        <div class="card-split">
            <div class="media-col">${videoHtml}${imgHtml}</div>
            <div class="ia-col">
                <div class="ia-section" id="ias-${jid}">${buildIaSection(jid, job)}</div>
            </div>
        </div>
        <div class="complice-wrap">
            <div class="complice-label">✋ Tu respuesta (Cómplice) — toca para cambiar</div>
            <div class="keys-row" id="keys-${jid}">${buildKeys(jid, merged, cols)}</div>
            <input class="complice-text" id="txt-${jid}" type="text"
                   value="${merged}" placeholder="O escribe aquí: ABCDA..."
                   oninput="onTextInput('${jid}', this)"
                   maxlength="50" autocomplete="off" spellcheck="false">
        </div>
        <div class="actions-row">
            <label class="check-mini">
                <input type="checkbox" id="img-${jid}">
                <span>🖼️ Imagen en pregunta</span>
            </label>
            <button class="btn-sm" onclick="resetToFusion('${jid}')" type="button">↺ Fusión</button>
            <button class="btn-sm btn-sm-red" onclick="clearAll('${jid}')" type="button">✕ Limpiar</button>
        </div>
        <button class="btn-send" id="send-${jid}" onclick="sendToMobile('${jid}')" type="button">
            <span class="send-label">
                <span class="send-label-main">✅ ENVIAR AL MÓVIL</span>
                <span class="send-label-sub">🔀 Fusión IA · edita arriba para cambiar</span>
            </span>
            <span class="send-prev" id="prev-${jid}">${merged || '---'}</span>
        </button>
    </div>`;
}

// Renderiza la fase actual del job en una sola línea legible.
// Para videos hace doble servicio: además de la fase, muestra cuántos
// OCRs llevamos y cuántos analyzers están corriendo.
function renderPhase(job) {
    const phase = job.phase || (job.has_video ? 'received' : 'analysis_running');
    const PHASE_LABELS = {
        'received':         '📥 RECIBIDO (video)',
        'ocr_running':      '🔍 OCR DEL VIDEO',
        'ocr_done':         '✅ OCR COMPLETO',
        'analysis_running': '🧠 ANALIZANDO (texto)',
        'analysis_done':    '✅ ANÁLISIS COMPLETO',
        'awaiting_review':  '✏️ ESPERANDO REVISIÓN',
        'done':             '✅ DONE',
        'error_ocr':        '❌ OCR FALLÓ',
        'error_pipeline':   '❌ PIPELINE FALLÓ',
    };
    let label = PHASE_LABELS[phase] || phase;

    // Detalle adicional según fase
    let extra = '';
    if(phase === 'ocr_running') {
        // Buscar la última entrada de phase_history para conocer progreso
        const hist = job.phase_history || [];
        for(let i = hist.length - 1; i >= 0; i--) {
            const h = hist[i];
            if(h.phase === 'ocr_running' && h.ocr_done !== undefined) {
                extra = ` (${h.ocr_done}/${h.ocr_total} listos)`;
                break;
            }
        }
    } else if(phase === 'ocr_done') {
        const hist = job.phase_history || [];
        for(let i = hist.length - 1; i >= 0; i--) {
            const h = hist[i];
            if(h.phase === 'ocr_done') {
                extra = ` · ${h.ocr_chars || 0} chars · ${h.ocr_ok_count}/${h.ocr_total} OK`;
                break;
            }
        }
    } else if(phase === 'analysis_running') {
        const ana = (job._providers || []).join(', ');
        if(ana) extra = ` [${ana}]`;
    }
    return `<b>FASE:</b> ${label}${extra}`;
}

// Catálogo de proveedores conocidos y su etiqueta humana en el panel.
// Mantener sincronizado con _PROVIDERS y _OCR_PROVIDERS en main.py.
const PROVIDER_LABELS = {
    // Analyzers (fase 2 en video, fase única en imagen)
    'gpt':        'GPT',
    'claude':     'Claude',
    'gemini':     'Gemini',
    'deepseek':   'DeepSeek',
    'mistral':    'Mistral',
    // OCRs (fase 1, sólo video)
    'qwen_ocr':      '🎬 Qwen OCR',
    'gemini_ocr':    '🎬 Gemini OCR',
    'kimi_ocr':      '🎬 Kimi OCR',
    'mimo_ocr':      '🎬 MiMo OCR',
    'anthropic_ocr': '🎬 Claude OCR',
    'openai_ocr':    '🎬 GPT-4o OCR',
    'nvidia_ocr':    '🎬 NVIDIA OCR',
};

function buildIaSection(jid, job) {
    const merged   = job.merged_answer || '';
    const expected = job.expected_questions || 0;
    const cols     = Math.max(expected, merged.length, 1);
    const resp     = job.responses || {};
    let html = '';

    // ── Fase 1: OCRs (sólo videos) ──────────────────────────────────────
    if(job.has_video) {
        const ocrResults = job.ocr_results || {};
        // Los OCRs que corren en paralelo en fase 1 (ver _OCR_PROVIDERS).
        // MiMo (Xiaomi) se añadió en el flujo full-modal; antes solo estaban los 3
        // primeros y MiMo no aparecía aunque hubiera key configurada. Claude OCR
        // se añadió posteriormente — usa la API de visión sobre un frame del video.
        const ocrProvs = ['qwen_ocr', 'gemini_ocr', 'kimi_ocr', 'mimo_ocr', 'anthropic_ocr', 'openai_ocr', 'nvidia_ocr'];
        // Mostrar header sólo si hay al menos info de un OCR esperable
        html += `<div class="ocr-header" style="font-size:10px;color:#888;
                       text-transform:uppercase;letter-spacing:1px;padding:6px 4px 2px">
                    📝 OCR del video (fase 1)
                 </div>`;
        for(const prov of ocrProvs) {
            html += buildOcrRow(jid, prov, PROVIDER_LABELS[prov] || prov, ocrResults[prov]);
        }
        // Fila especial: el OCR fusionado por consenso (server-side). Muestra
        // K preguntas únicas y permite ver el texto canónico que se envió a los
        // analyzers. Si la fusión falló, aparece en amarillo indicando fallback.
        html += buildOcrFusionRow(jid, job);
        html += `<div class="ocr-header" style="font-size:10px;color:#888;
                       text-transform:uppercase;letter-spacing:1px;padding:8px 4px 2px">
                    🧠 Analyzers (fase 2 · sobre OCR fusionado)
                 </div>`;
    }

    // ── Fase 2 (o única): analyzers ────────────────────────────────────
    // Usamos job._providers para conocer EXACTAMENTE qué analyzers corren
    // en este job (modo IMAGEN excluye deepseek/mistral, modo VIDEO los incluye).
    // Si por algún motivo falta, hacemos fallback al catálogo completo.
    const provs = (job._providers && job._providers.length)
        ? job._providers
        : ['gpt','claude','gemini','deepseek','mistral'];
    for(const prov of provs) {
        const label = PROVIDER_LABELS[prov] || prov.toUpperCase();
        html += buildIaRow(jid, prov, label, resp[prov], merged, cols);
    }
    // NB: la fila de Fusión ya NO va aquí — es la respuesta que se envía por
    // defecto si el cómplice no edita, así que está integrada visualmente en
    // el propio botón "ENVIAR AL MÓVIL" (mira buildCard).
    return html;
}

// Fila para un OCR provider. Estados: waiting (aún no corrió), ok (con chars),
// error (con motivo). NO tiene botones de envío — es solo informativo.
function buildOcrRow(jid, prov, label, r) {
    const id = `ocr-${prov}-${jid}`;
    if(!r) {
        return `<div class="ia-row" id="${id}">
                    <span class="ia-name">${label}</span>
                    <span class="ia-wait">⏳ esperando OCR...</span>
                </div>`;
    }
    const ms = r.ms ? `<small>${ms_fmt(r.ms)}</small>` : '';
    if(r.ok) {
        // chars/preview: el backend NUEVO precomputa r.chars + r.preview en el
        // slim (/api/jobs strippea r.text para no inflar payload). Si el backend
        // está corriendo una versión vieja (pre-precompute), `r.chars` no llega
        // y `r.text` está strippeado → ambos undefined.
        let charsLabel, charsVal;
        if(r.chars != null) {
            charsVal = r.chars;
            charsLabel = `${charsVal} chars`;
        } else if(r.text != null) {
            charsVal = r.text.length;
            charsLabel = `${charsVal} chars`;
        } else {
            charsVal = null;
            charsLabel = 'OK';
        }
        if(r.text) {
            window._OCR_TEXTS = window._OCR_TEXTS || {};
            window._OCR_TEXTS[`${jid}|${prov}`] = r.text;
        }
        const previewSrc = r.text || r.preview || r.preview_diag || '';
        const preview = previewSrc.slice(0, 60).replace(/[<>&"]/g, '');
        // Calidad: el backend marca quality=empty | no_legible | ok según el
        // texto devuelto. Si no llega quality (backend viejo o cubrir caso),
        // inferimos por chars/preview. NUNCA mostramos ✅ verde brillante para
        // 0 chars o NO_LEGIBLE — eso era engañoso (parecía éxito real).
        let quality = r.quality;
        if(!quality) {
            const previewUpper = (previewSrc || '').trim().toUpperCase();
            if(charsVal === 0) quality = 'empty';
            else if(previewUpper === 'NO_LEGIBLE' || previewUpper.startsWith('NO_LEGIBLE')) quality = 'no_legible';
            else quality = 'ok';
        }
        let badge, badgeColor;
        if(quality === 'empty') {
            badge = `⚠️ vacío`;
            badgeColor = '#e0c060';   // ámbar — preocupante
        } else if(quality === 'no_legible') {
            badge = `⓵ NO_LEGIBLE`;
            badgeColor = '#888';      // gris — informativo
        } else {
            badge = `✅ ${charsLabel}`;
            badgeColor = '#5d8';      // verde — éxito real
        }
        return `<div class="ia-row" id="${id}" style="align-items:center">
                    <span class="ia-name">${label}${ms}</span>
                    <span style="color:${badgeColor};font-weight:600">${badge}</span>
                    <span style="color:#aaa;font-size:11px;font-family:monospace;
                                 overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                                 max-width:280px;flex:1">
                        ${preview}${(charsVal != null && charsVal > 60) ? '…' : ''}
                    </span>
                    <button onclick="showOcrText('${jid}','${prov}','${label.replace(/[\\'`]/g,'')}')"
                            style="background:#0d6efd;color:#fff;border:none;border-radius:4px;
                                   padding:3px 10px;font-size:11px;font-weight:700;cursor:pointer"
                            title="Ver transcripción completa">📜 Ver</button>
                </div>`;
    } else {
        const err = (r.error || 'error').slice(0, 80);
        return `<div class="ia-row" id="${id}">
                    <span class="ia-name">${label}${ms}</span>
                    <span class="ia-err" title="${err}">❌ ${err}</span>
                </div>`;
    }
}

// Fila para el OCR FUSIONADO (resultado del consenso server-side de las N OCRs).
// Estados posibles:
//   - waiting:   aún no hay fusión calculada (fase OCR sin terminar)
//   - ok+used:   fusión exitosa y se usó para los analyzers (verde)
//   - ok+fallback: fusión calculada pero no se usó (fallback a <readings> crudo, amarillo)
//   - empty:     ningún OCR parseable, fusión vacía (gris)
function buildOcrFusionRow(jid, job) {
    const id = `ocr-fusion-${jid}`;
    const fusedText = job.ocr_fused_text || '';
    const stats     = job.ocr_fusion_stats || {};
    const used      = job.ocr_fusion_used === true;
    const ocrResults = job.ocr_results || {};
    // Si ningún OCR ha terminado aún, mostramos estado "esperando"
    const anyOcrDone = Object.values(ocrResults).some(r => r && (r.ok || r.error));
    if(!anyOcrDone) {
        return `<div class="ia-row" id="${id}" style="background:#16181c;border-left:3px solid #444">
                    <span class="ia-name">🧬 Fusión OCR</span>
                    <span class="ia-wait">⏳ esperando OCRs...</span>
                </div>`;
    }
    // Caso: fusión hecha y usada → verde brillante
    if(fusedText && used) {
        // chars REAL: el backend trunca ocr_fused_text a 200 chars en el listado
        // y manda el len original en ocr_fused_text_full_chars. Usar este último
        // para no mostrar siempre "200 chars" cuando el original era más largo.
        const chars = (job.ocr_fused_text_full_chars != null)
                        ? job.ocr_fused_text_full_chars
                        : fusedText.length;
        const k = stats.clusters || 0;
        const parsedMap = stats.parsed_per_ocr || {};
        const parsedList = Object.entries(parsedMap)
            .map(([p,n]) => `${p.replace('_ocr','')}:${n}`).join(' ');
        // Cachear SOLO la versión completa (el listado slim trunca a 200 chars y
        // marca _ocr_fused_truncated=true). Si cacheamos el truncado, showOcrText
        // lo lee de cache y NO lazy-fetchea el completo → popup queda cortado.
        // El bug histórico: aquí escribía sin guard → popup mostraba "200 chars".
        window._OCR_TEXTS = window._OCR_TEXTS || {};
        if(!job._ocr_fused_truncated) {
            window._OCR_TEXTS[`${jid}|fusion`] = fusedText;
        }
        const preview = fusedText.slice(0, 60).replace(/[<>&"]/g, '');
        return `<div class="ia-row" id="${id}" style="background:#0d1a0d;border-left:3px solid #5d8;align-items:center">
                    <span class="ia-name" style="color:#7df097">🧬 Fusión OCR <small style="color:#888;font-weight:400">→ analyzers</small></span>
                    <span style="color:#5d8;font-weight:600">✅ ${k} preguntas · ${chars} chars</span>
                    <span style="color:#aaa;font-size:11px;font-family:monospace;
                                 overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                                 max-width:240px;flex:1" title="parsed: ${parsedList}">
                        ${preview}${chars > 60 ? '…' : ''}
                    </span>
                    <button onclick="showOcrText('${jid}','fusion','Fusión OCR (consenso)')"
                            style="background:#198754;color:#fff;border:none;border-radius:4px;
                                   padding:3px 10px;font-size:11px;font-weight:700;cursor:pointer"
                            title="Ver transcripción fusionada completa">📜 Ver</button>
                </div>`;
    }
    // Caso: fusión calculada pero NO usada → fallback a <readings>, color amarillo
    if(fusedText && !used) {
        const chars = (job.ocr_fused_text_full_chars != null)
                        ? job.ocr_fused_text_full_chars
                        : fusedText.length;
        const k = stats.clusters || 0;
        // Mismo guard que la rama "used": NO cachear texto truncado o el popup se
        // queda con la versión cortada y NO hace lazy-fetch.
        window._OCR_TEXTS = window._OCR_TEXTS || {};
        if(!job._ocr_fused_truncated) {
            window._OCR_TEXTS[`${jid}|fusion`] = fusedText;
        }
        return `<div class="ia-row" id="${id}" style="background:#1a1a0d;border-left:3px solid #d4a017;align-items:center">
                    <span class="ia-name" style="color:#e0c060">🧬 Fusión OCR</span>
                    <span style="color:#e0c060;font-weight:600">⚠️ ${k} preguntas · no usada (fallback)</span>
                    <button onclick="showOcrText('${jid}','fusion','Fusión OCR (calculada pero no usada)')"
                            style="background:#d4a017;color:#000;border:none;border-radius:4px;
                                   padding:3px 10px;font-size:11px;font-weight:700;cursor:pointer">📜 Ver</button>
                </div>`;
    }
    // Caso: fusión vacía o con error → gris
    const errInfo = stats.error ? `: ${stats.error}` : '';
    return `<div class="ia-row" id="${id}" style="background:#1a0d0d;border-left:3px solid #666">
                <span class="ia-name" style="color:#999">🧬 Fusión OCR</span>
                <span style="color:#aaa">⚪ sin fusión (analyzers reciben &lt;readings&gt; crudo)${errInfo}</span>
            </div>`;
}

// Modal genérico para mostrar texto completo (OCR transcrito o raw de IA).
// Se monta lazy la primera vez; lo reusan showOcrText y showRawResponse.
function showFullText(title, meta, txt) {
    let modal = document.getElementById('ocr-modal');
    if(!modal) {
        modal = document.createElement('div');
        modal.id = 'ocr-modal';
        modal.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.85);'
                            + 'z-index:9999;display:none;align-items:center;justify-content:center;padding:24px';
        modal.innerHTML = `
            <div style="background:#1a1a1a;border:1px solid #333;border-radius:10px;
                        max-width:900px;max-height:85vh;width:100%;display:flex;flex-direction:column;overflow:hidden">
                <div style="display:flex;align-items:center;justify-content:space-between;
                            padding:12px 16px;background:#0d0d0d;border-bottom:1px solid #2a2a2a">
                    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
                        <span id="ocr-modal-title" style="font-weight:700;color:#fff">📜 Transcripción</span>
                        <span id="ocr-modal-meta" style="font-size:11px;color:#888"></span>
                    </div>
                    <div style="display:flex;gap:6px">
                        <button onclick="copyOcrText()" style="background:#0d6efd;color:#fff;border:none;
                                border-radius:4px;padding:5px 10px;font-size:11px;cursor:pointer">📋 Copiar</button>
                        <button onclick="document.getElementById('ocr-modal').style.display='none'"
                                style="background:#dc3545;color:#fff;border:none;border-radius:4px;
                                       padding:5px 10px;font-size:11px;cursor:pointer">✕ Cerrar</button>
                    </div>
                </div>
                <textarea id="ocr-modal-text" readonly
                          style="flex:1;background:#0d0d0d;color:#e7e7e7;border:none;padding:14px;
                                 font-family:'SF Mono','Consolas',monospace;font-size:12px;line-height:1.5;
                                 resize:none;outline:none;min-height:300px"></textarea>
            </div>`;
        document.body.appendChild(modal);
        // Cerrar al pulsar fondo (no el contenido)
        modal.addEventListener('click', e => { if(e.target === modal) modal.style.display='none'; });
    }
    document.getElementById('ocr-modal-title').textContent = title;
    document.getElementById('ocr-modal-meta').textContent  = meta;
    document.getElementById('ocr-modal-text').value        = txt || '(vacío)';
    modal.style.display = 'flex';
}

// Variante HTML del modal (en lugar de textarea, un div con innerHTML). Usado
// para vistas estructuradas con enlaces clickables (búsquedas Tavily, etc.).
// Modal separado para no compartir DOM con showFullText (que usa textarea).
function showFullTextHtml(title, meta, html) {
    let modal = document.getElementById('html-modal');
    if(!modal) {
        modal = document.createElement('div');
        modal.id = 'html-modal';
        modal.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.85);'
                            + 'z-index:9999;display:none;align-items:center;justify-content:center;padding:24px';
        modal.innerHTML = `
            <div style="background:#1a1a1a;border:1px solid #333;border-radius:10px;
                        max-width:900px;max-height:85vh;width:100%;display:flex;flex-direction:column;overflow:hidden">
                <div style="display:flex;align-items:center;justify-content:space-between;
                            padding:12px 16px;background:#0d0d0d;border-bottom:1px solid #2a2a2a">
                    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
                        <span id="html-modal-title" style="font-weight:700;color:#fff"></span>
                        <span id="html-modal-meta" style="font-size:11px;color:#888"></span>
                    </div>
                    <button onclick="document.getElementById('html-modal').style.display='none'"
                            style="background:#dc3545;color:#fff;border:none;border-radius:4px;
                                   padding:5px 10px;font-size:11px;cursor:pointer">✕ Cerrar</button>
                </div>
                <div id="html-modal-body" style="flex:1;overflow:auto;padding:14px;background:#0d0d0d;
                                                  color:#e7e7e7;font-family:'Segoe UI',system-ui,sans-serif;
                                                  font-size:13px;line-height:1.5"></div>
            </div>`;
        document.body.appendChild(modal);
        modal.addEventListener('click', e => { if(e.target === modal) modal.style.display='none'; });
    }
    document.getElementById('html-modal-title').textContent = title;
    document.getElementById('html-modal-meta').textContent  = meta;
    document.getElementById('html-modal-body').innerHTML    = html || '(vacío)';
    modal.style.display = 'flex';
}

// Cache helper: pide /api/partial/{jid} y devuelve el job completo.
// _PARTIAL_FETCHES dedupea peticiones concurrentes (si el usuario abre dos
// popups del mismo job en rápida sucesión, la 2ª espera al promise de la 1ª).
window._PARTIAL_FETCHES = window._PARTIAL_FETCHES || {};
async function fetchPartialJob(jid) {
    if(window._PARTIAL_FETCHES[jid]) return window._PARTIAL_FETCHES[jid];
    const p = (async () => {
        const r = await fetch('/api/partial/' + encodeURIComponent(jid)
                              + '?key=' + encodeURIComponent(KEY),
                              { headers: { 'X-Api-Key': KEY } });
        if(!r.ok) {
            let detail = '';
            try { detail = (await r.json()).detail || ''; } catch(_) {}
            throw new Error('HTTP ' + r.status + (detail ? ' · ' + detail : ''));
        }
        return await r.json();
    })();
    window._PARTIAL_FETCHES[jid] = p;
    // Liberamos el slot tras unos segundos para futuras llamadas (no cacheamos
    // el JSON entero — solo el promise en vuelo). Las extracciones individuales
    // sí se cachean en _OCR_TEXTS / _RAW_RESPONSES.
    p.finally(() => { setTimeout(() => { delete window._PARTIAL_FETCHES[jid]; }, 3000); });
    return p;
}

// Modal con la transcripción OCR completa. Si la cache local no la tiene
// (caso normal en el listado slim — el backend strippea ocr_results[*].text
// y trunca ocr_fused_text a 200 chars para no inflar el polling), hacemos
// lazy-fetch a /api/partial/{jid} y la cacheamos para próximos clicks.
async function showOcrText(jid, prov, label) {
    const key = `${jid}|${prov}`;
    let txt = (window._OCR_TEXTS && window._OCR_TEXTS[key]) || '';
    if(!txt) {
        showFullText(`📜 ${label}`,
                     `cargando... · job ${jid.slice(0,8)}`,
                     '⏳ Cargando transcripción completa desde el servidor...');
        try {
            const j = await fetchPartialJob(jid);
            if(prov === 'fusion') {
                txt = j.ocr_fused_text || '';
            } else {
                txt = ((j.ocr_results || {})[prov] || {}).text || '';
            }
            if(txt) {
                window._OCR_TEXTS = window._OCR_TEXTS || {};
                window._OCR_TEXTS[key] = txt;
            }
        } catch(e) {
            showFullText(`📜 ${label}`,
                         `error · job ${jid.slice(0,8)}`,
                         '⚠ No se pudo cargar la transcripción: ' + (e.message || 'error desconocido'));
            return;
        }
    }
    showFullText(`📜 ${label}`,
                 `${txt.length} chars · job ${jid.slice(0,8)}`,
                 txt);
}

// Modal con el bloque <internet_context> EXACTO que se inyectó al prompt del
// razonador. Si en el listado slim solo llegó truncado (>200 chars), hace
// lazy-fetch a /api/partial para conseguir el bloque completo.
async function showTavilyBlock(jid) {
    let txt = (window._TAVILY_BLOCKS && window._TAVILY_BLOCKS[jid]) || '';
    if(!txt) {
        showFullText(`🌐 Tavily · bloque inyectado`,
                     `cargando... · job ${jid.slice(0,8)}`,
                     '⏳ Cargando bloque <internet_context> completo desde el servidor...');
        try {
            const j = await fetchPartialJob(jid);
            txt = j.tavily_block || '';
            if(txt) {
                window._TAVILY_BLOCKS = window._TAVILY_BLOCKS || {};
                window._TAVILY_BLOCKS[jid] = txt;
            }
        } catch(e) {
            showFullText(`🌐 Tavily · bloque inyectado`,
                         `error · job ${jid.slice(0,8)}`,
                         '⚠ No se pudo cargar el bloque: ' + (e.message || 'error'));
            return;
        }
    }
    if(!txt) {
        showFullText(`🌐 Tavily · bloque inyectado`,
                     `vacío · job ${jid.slice(0,8)}`,
                     '(Tavily no inyectó nada — sin resultados utilizables o paso omitido)');
        return;
    }
    showFullText(`🌐 Tavily · bloque inyectado al prompt`,
                 `${txt.length} chars · job ${jid.slice(0,8)}`,
                 txt);
}

// Modal con la lista detallada de búsquedas Tavily: una sección por pregunta,
// query enviada, fuentes encontradas (title + url + snippet recortado).
function showTavilyQueries(jid) {
    const stats = (window._TAVILY_STATS && window._TAVILY_STATS[jid]) || null;
    if(!stats || !Array.isArray(stats.queries) || !stats.queries.length) {
        showFullText('🔎 Tavily · búsquedas',
                     `vacío · job ${jid.slice(0,8)}`,
                     '(no hay queries guardadas para este job)');
        return;
    }
    const escapeHtml = s => String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');  // A10: incluir comilla simple
    const blocks = stats.queries.map((q, i) => {
        const tag = q.ok
            ? `<span style="color:#0ea5e9">✓ ${q.n_sources} fuentes · ${ms_fmt(q.ms||0)}</span>`
            : `<span style="color:#dc3545">❌ ${escapeHtml(q.error || 'sin respuesta')}</span>`;
        const numLabel = (q.num != null) ? `P${q.num}` : `P?${i+1}`;
        const sources = (q.sources || []).map(s => {
            const title = escapeHtml((s.title || '(sin título)'));
            const url   = escapeHtml(s.url || '');
            const snip  = escapeHtml(s.snippet || '');
            return `<div style="margin:4px 0 4px 12px;padding:4px 8px;background:#0a0a0a;
                                 border-left:2px solid #0ea5e9;border-radius:3px">
                        <div style="color:#9cf;font-weight:600">${title}</div>
                        <div><a href="${url}" target="_blank" rel="noopener"
                                style="color:#0ea5e9;font-size:11px;text-decoration:none">${url}</a></div>
                        ${snip ? `<div style="color:#bbb;font-size:11px;margin-top:3px">${snip}</div>` : ''}
                    </div>`;
        }).join('');
        return `<div style="margin-bottom:10px;padding:6px 8px;background:#101010;border:1px solid #222;border-radius:5px">
                    <div style="display:flex;gap:8px;align-items:center;margin-bottom:4px">
                        <span style="background:#0ea5e9;color:#000;font-weight:700;padding:1px 6px;
                                     border-radius:3px;font-size:11px">${numLabel}</span>
                        ${tag}
                    </div>
                    <div style="color:#aaa;font-size:11px;font-family:monospace;margin-bottom:4px">
                        🔎 ${escapeHtml(q.query || '(query vacía)')}
                    </div>
                    ${sources || '<div style="color:#666;font-size:11px;font-style:italic">(sin fuentes)</div>'}
                </div>`;
    }).join('');
    showFullTextHtml('🔎 Tavily · búsquedas por pregunta',
                     `${stats.queries.length} preguntas · ${stats.n_with_sources || 0} con fuentes · ${ms_fmt(stats.elapsed_ms||0)}`,
                     blocks);
}

// Modal con la respuesta cruda completa de UN analyzer (incluye razonamiento,
// citas inline, texto antes de la extracción de letras, etc.). Cargado desde
// window._RAW_RESPONSES para no inflar el DOM con cada raw de cada IA del top-30.
// Igual que showOcrText: lazy-fetch desde /api/partial si la cache está vacía
// (el listado slim strippea responses[*].raw).
async function showRawResponse(jid, prov, label) {
    const key = `${jid}|${prov}`;
    let txt = (window._RAW_RESPONSES && window._RAW_RESPONSES[key]) || '';
    if(!txt) {
        showFullText(`💬 ${label} · respuesta completa`,
                     `cargando... · job ${jid.slice(0,8)}`,
                     '⏳ Cargando respuesta completa desde el servidor...');
        try {
            const j = await fetchPartialJob(jid);
            txt = ((j.responses || {})[prov] || {}).raw || '';
            if(txt) {
                window._RAW_RESPONSES = window._RAW_RESPONSES || {};
                window._RAW_RESPONSES[key] = txt;
            }
        } catch(e) {
            showFullText(`💬 ${label} · respuesta completa`,
                         `error · job ${jid.slice(0,8)}`,
                         '⚠ No se pudo cargar la respuesta: ' + (e.message || 'error desconocido'));
            return;
        }
    }
    showFullText(`💬 ${label} · respuesta completa`,
                 `${txt.length} chars · job ${jid.slice(0,8)}`,
                 txt);
}

function copyOcrText() {
    const ta = document.getElementById('ocr-modal-text');
    if(!ta) return;
    ta.select();
    try { navigator.clipboard.writeText(ta.value); toast('📋 Copiado'); }
    catch(_) { document.execCommand('copy'); toast('📋 Copiado'); }
}

function buildIaRow(jid, prov, label, r, merged, cols) {
    // Las filas individuales de IA NO envían — solo cargan al editor del
    // cómplice (botón 📋). El único botón de envío REAL es el verde grande
    // "✅ ENVIAR AL MÓVIL" al fondo del card.
    const id = `ia-${prov}-${jid}`;
    const ms = r && r.ms ? `<small>${ms_fmt(r.ms)}</small>` : '';
    let body = '';
    if(!r || r.status === 'waiting' || r.status === 'pending') {
        body = `<span class="ia-wait">⏳ esperando...</span>
                <button class="btn-use" disabled>📋</button>`;
    } else if(r.status === 'processing') {
        body = `<span class="ia-proc">⚡ procesando...</span>
                <button class="btn-use" disabled>📋</button>`;
    } else if(r.status === 'no_key') {
        // Sin key configurada: NO es un fallo, simplemente no participa.
        // Estilo gris/neutro para no confundir con un error real.
        body = `<span style="color:#888;font-size:11px;font-style:italic">
                    🔑 sin key configurada
                </span>
                <button class="btn-use" disabled>📋</button>`;
    } else if(r.ok && r.answer) {
        const ans = r.answer.replace(/'/g, "\\'");
        body = `<div class="cells">${buildCells(r.answer, cols, merged)}</div>
                <button class="btn-use" title="Cargar esta respuesta al editor"
                        onclick="useAns('${jid}','${ans}')">📋</button>`;
    } else {
        const err = ((r && r.error) || 'error').slice(0, 60);
        body = `<span class="ia-err" title="${err}">❌ ${err}</span>
                <button class="btn-use" disabled>📋</button>`;
    }
    return `<div class="ia-row" id="${id}">
                <span class="ia-name">${label}${ms}</span>
                ${body}
            </div>`;
}

function buildFusionRow(jid, merged, cols) {
    let body;
    if(merged) {
        const m = merged.replace(/'/g, "\\'");
        body = `<div class="cells">${buildCells(merged, cols, merged)}</div>
                <button class="btn-use"   title="Copiar al editor"
                        onclick="useAns('${jid}','${m}')">📋</button>
                <button class="btn-quick" title="Enviar fusión"
                        onclick="sendDirect('${jid}','${m}','Fusión')">📤</button>`;
    } else {
        body = `<span class="ia-wait">Esperando IAs...</span>
                <button class="btn-use"   disabled>📋</button>
                <button class="btn-quick" disabled>📤</button>`;
    }
    return `<div class="ia-row ia-fusion" id="ia-fusion-${jid}">
                <span class="ia-name">🔀 Fusión</span>
                ${body}
            </div>`;
}

function ms_fmt(ms) {
    if(ms < 1000) return ms + 'ms';
    return (ms/1000).toFixed(1) + 's';
}

function buildCells(ans, cols, merged) {
    const n = Math.max(cols, ans ? ans.length : 0);
    let html = '';
    for(let i = 0; i < n; i++) {
        const ch = (ans && i < ans.length) ? ans[i] : '?';
        const cls = 'c-' + (ch.match(/[ABCDX]/) ? ch : 'q');
        const diff = (merged && i < merged.length && merged[i] !== ch) ? ' diff' : '';
        html += `<span class="cell ${cls}${diff}">${ch}</span>`;
    }
    return html;
}

function buildKeys(jid, ans, cols) {
    let html = '';
    for(let i = 0; i < cols; i++) {
        const ch = (ans && i < ans.length) ? ans[i] : 'X';
        html += `<button type="button" class="key k-${ch}" data-i="${i}"
                         onclick="rotKey(this,'${jid}')">
                     <span class="key-num">${i+1}</span>
                     <span class="key-letter">${ch}</span>
                 </button>`;
    }
    return html;
}

// ── Actualizar tarjeta en sitio (sin destruir inputs) ────────────────────────
function updateCard(card, job) {
    const jid    = job.id;
    const merged = job.merged_answer || '';
    const expq   = job.expected_questions || 0;
    const cols   = Math.max(expq, merged.length, 1);

    // Cambio de estado
    if(card.dataset.status !== job.status) {
        card.dataset.status = job.status;
        const badge = document.getElementById('badge-' + jid);
        if(job.status === 'awaiting_review') {
            card.className = 'card card-rev';
            if(badge) { badge.className = 'badge b-rev'; badge.textContent = '✏️ REVISAR'; }
            // Actualizar deadline en contador
            const cd = document.getElementById('cd-' + jid);
            if(cd && job.review_deadline) {
                cd.dataset.deadline = job.review_deadline;
                cd.dataset.total    = job.review_timeout_seconds || 90;
            }
        }
    }

    // Actualizar la fase (siempre, incluso si status no cambió: phase es más granular)
    const phaseEl = document.getElementById('phase-' + jid);
    if(phaseEl) {
        const newPhase = renderPhase(job);
        if(phaseEl.innerHTML !== newPhase) phaseEl.innerHTML = newPhase;
    }

    // Actualizar filas de IA en sitio. Estrategia: si el conjunto de providers
    // efectivos ya está renderizado, actualizamos cada fila individualmente
    // (no destruimos los inputs). Si NO coincide (caso raro: server cambia
    // la lista por reinicio o cambio de modo), re-render completo de la sección.
    const resp = job.responses || {};
    const provs = (job._providers && job._providers.length)
        ? job._providers
        : ['gpt','claude','gemini','deepseek','mistral'];

    // ¿Las filas existentes coinciden con los providers actuales?
    let needFullRebuild = false;
    for(const prov of provs) {
        if(!document.getElementById(`ia-${prov}-${jid}`)) {
            needFullRebuild = true; break;
        }
    }
    if(needFullRebuild) {
        const iasEl = document.getElementById('ias-' + jid);
        if(iasEl) iasEl.innerHTML = buildIaSection(jid, job);
    } else {
        // OCR rows (solo videos): actualizar si existen
        if(job.has_video) {
            const ocrResults = job.ocr_results || {};
            for(const prov of ['qwen_ocr', 'gemini_ocr', 'kimi_ocr', 'mimo_ocr', 'anthropic_ocr', 'openai_ocr', 'nvidia_ocr']) {
                const rowEl = document.getElementById(`ocr-${prov}-${jid}`);
                if(!rowEl) continue;
                rowEl.outerHTML = buildOcrRow(jid, prov, PROVIDER_LABELS[prov] || prov, ocrResults[prov]);
            }
            // Refrescar también la fila de OCR fusionado (puede haber llegado
            // después de las OCRs individuales).
            const fusRowEl = document.getElementById(`ocr-fusion-${jid}`);
            if(fusRowEl) fusRowEl.outerHTML = buildOcrFusionRow(jid, job);
        }
        // Analyzer rows
        for(const prov of provs) {
            const rowEl = document.getElementById(`ia-${prov}-${jid}`);
            if(!rowEl) continue;
            const label = PROVIDER_LABELS[prov] || prov.toUpperCase();
            rowEl.outerHTML = buildIaRow(jid, prov, label, resp[prov], merged, cols);
        }
        // (Fila de Fusión eliminada — el preview vive ahora en el botón de envío.)
    }

    // Si el Cómplice no ha editado → sincronizar teclas con fusión
    const state = _STATE[jid] || (_STATE[jid] = {dirty: false});
    if(!state.dirty && merged && merged !== card.dataset.merged) {
        card.dataset.merged = merged;
        _setKeys(jid, merged);
        const txt = document.getElementById('txt-' + jid);
        if(txt && document.activeElement !== txt) txt.value = merged;
        _updPreview(jid);
    }

    // Ampliar teclas si llegaron más preguntas que las que había
    const currentCols = document.querySelectorAll(`#keys-${jid} .key`).length;
    if(cols > currentCols) {
        const keysEl = document.getElementById('keys-' + jid);
        if(keysEl) {
            const cur = _getAns(jid);
            keysEl.innerHTML = buildKeys(jid, cur.padEnd(cols, 'X'), cols);
            card.dataset.cols = cols;
        }
    }
}

// ── Countdown ─────────────────────────────────────────────────────────────────
function clsP(p) { return p < 0.25 ? 'cd-urg' : p < 0.5 ? 'cd-warn' : 'cd-ok'; }

function ticks() {
    const now = Date.now() / 1000;
    document.querySelectorAll('.cd[data-deadline]').forEach(el => {
        const dl  = parseFloat(el.dataset.deadline);
        const tot = parseFloat(el.dataset.total) || 90;
        if(!dl || dl <= 0) return;
        const left = Math.round(dl - now);
        if(left <= 0) { el.textContent = '⚡ auto'; el.className = 'cd cd-urg'; }
        else { el.textContent = '⏱ ' + left + 's'; el.className = 'cd ' + clsP(left/tot); }
    });
}

// ── Keys (tap para rotar A→B→C→D→X) ─────────────────────────────────────────
function rotKey(btn, jid) {
    const lEl = btn.querySelector('.key-letter');
    const next = _LET[(_LET.indexOf(lEl.textContent) + 1) % _LET.length];
    lEl.textContent = next;
    btn.className = btn.className.replace(/k-[ABCDX]/, 'k-' + next);
    _markDirty(jid);
    _syncTxt(jid);
    _updPreview(jid);
}

function _getAns(jid) {
    return Array.from(document.querySelectorAll(`#keys-${jid} .key .key-letter`))
               .map(k => k.textContent).join('');
}

function _setKeys(jid, ans) {
    const keys = document.querySelectorAll(`#keys-${jid} .key`);
    const n    = keys.length;
    const s    = (ans || '').padEnd(n, 'X').slice(0, n);
    keys.forEach((k, i) => {
        const ch = s[i] || 'X';
        k.querySelector('.key-letter').textContent = ch;
        k.className = k.className.replace(/k-[ABCDX]/, 'k-' + ch);
    });
    _updPreview(jid);
}

function _syncTxt(jid) {
    const t = document.getElementById('txt-' + jid);
    if(t) t.value = _getAns(jid);
}

function onTextInput(jid, input) {
    const v = input.value.toUpperCase().replace(/[^ABCDX]/g, '');
    input.value = v;
    _setKeys(jid, v);
    _markDirty(jid);
    _updPreview(jid);
}

function _updPreview(jid) {
    const p = document.getElementById('prev-' + jid);
    if(p) p.textContent = _getAns(jid) || '---';
}

function _markDirty(jid) { (_STATE[jid] || (_STATE[jid] = {})).dirty = true; }

// ── Usar / Reset / Clear ──────────────────────────────────────────────────────
function useAns(jid, ans) {
    _setKeys(jid, ans);
    const t = document.getElementById('txt-' + jid);
    if(t) t.value = ans;
    _markDirty(jid);
    _updPreview(jid);
    toast('Copiado: ' + ans);
}

function resetToFusion(jid) {
    const card   = document.getElementById('card-' + jid);
    const merged = card ? card.dataset.merged : '';
    _setKeys(jid, merged);
    const t = document.getElementById('txt-' + jid);
    if(t) t.value = merged;
    (_STATE[jid] || (_STATE[jid] = {})).dirty = false;
    _updPreview(jid);
}

function clearAll(jid) {
    const card = document.getElementById('card-' + jid);
    const cols = parseInt(card?.dataset.cols || '1', 10);
    const blank = 'X'.repeat(Math.max(1, cols));
    _setKeys(jid, blank);
    const t = document.getElementById('txt-' + jid);
    if(t) t.value = blank;
    _markDirty(jid);
    _updPreview(jid);
}

// ── Enviar DIRECTO una respuesta concreta (sin pasar por el editor) ─────────
async function sendDirect(jid, ans, label) {
    if(!ans) return;
    const card  = document.getElementById('card-' + jid);
    const imgEl = document.getElementById('img-' + jid);
    try {
        const r = await fetch('/result/' + jid, {
            method:  'PATCH',
            headers: {'Content-Type': 'application/json', 'X-Api-Key': KEY},
            body:    JSON.stringify({answer: ans, has_image: imgEl ? imgEl.checked : false})
        });
        if(!r.ok) throw new Error((await r.json()).detail || r.status);
        toast('✅ ' + label + ': ' + ans);
        if(card) { card.style.transition='opacity .35s'; card.style.opacity='0';
                    setTimeout(()=>card.remove(), 380); }
    } catch(e) {
        toast('❌ ' + e.message);
    }
}

// ── Enviar al móvil ───────────────────────────────────────────────────────────
async function sendToMobile(jid) {
    const ans = _getAns(jid);
    if(!ans || ans === '' ) { toast('No hay respuesta'); return; }
    const imgEl = document.getElementById('img-' + jid);
    const btn   = document.getElementById('send-' + jid);
    btn.disabled = true;
    btn.querySelector('span').textContent = 'ENVIANDO...';
    try {
        const r = await fetch('/result/' + jid, {
            method:  'PATCH',
            headers: {'Content-Type': 'application/json', 'X-Api-Key': KEY},
            body:    JSON.stringify({answer: ans, has_image: imgEl ? imgEl.checked : false})
        });
        if(!r.ok) throw new Error((await r.json()).detail || r.status);
        toast('✅ ' + ans + (imgEl?.checked ? ' 🖼️' : '') + ' enviado');
        const card = document.getElementById('card-' + jid);
        if(card) { card.style.transition='opacity .35s'; card.style.opacity='0';
                    setTimeout(()=>card.remove(), 380); }
    } catch(e) {
        btn.disabled = false;
        btn.querySelector('span').textContent = '✅ ENVIAR AL MÓVIL';
        alert('Error: ' + e.message);
    }
}

// ── Config: cargar valores actuales al abrir ────────────────────────────────
// ── Pie chart de pesos en la fusión ─────────────────────────────────────────
// Dibuja un donut SVG donde cada porción es proporcional al peso de cada IA en
// la votación. Se llama al cargar la config (loadConfig) y al mover cualquier
// slider (oninput). Si todos los pesos son 0 muestra placeholder.
const _WEIGHT_ITEMS = [
    { id:'cfg_w_ant', name:'Anthropic', color:'#3ddc84', emoji:'🟢' },
    { id:'cfg_w_oai', name:'OpenAI',    color:'#4d9eff', emoji:'🔵' },
    { id:'cfg_w_gem', name:'Gemini',    color:'#ffb066', emoji:'🟠' },
    { id:'cfg_w_dsk', name:'DeepSeek',  color:'#a855f7', emoji:'🟣' },
    { id:'cfg_w_mst', name:'Mistral',   color:'#fb7185', emoji:'🌶️' },
    { id:'cfg_w_mim', name:'MiMo',      color:'#ff6b00', emoji:'🤖' },
];
function renderWeightsPie() {
    const svg = document.getElementById('weights-pie-svg');
    const legend = document.getElementById('weights-pie-legend');
    if(!svg) return;
    const items = _WEIGHT_ITEMS.map(it => ({
        ...it,
        w: parseInt(document.getElementById(it.id)?.value) || 0
    }));
    const total = items.reduce((s, it) => s + it.w, 0);
    const cx = 70, cy = 70, rOuter = 60, rInner = 28;

    if(total === 0) {
        svg.innerHTML =
            `<circle cx="${cx}" cy="${cy}" r="${rOuter}" fill="#1a1a1a" stroke="#333"/>
             <text x="${cx}" y="${cy-4}" text-anchor="middle" font-size="10" fill="#666">sin pesos</text>
             <text x="${cx}" y="${cy+9}" text-anchor="middle" font-size="9" fill="#555">(todos a 0)</text>`;
        if(legend) legend.innerHTML = '<span style="color:#666">Sube algún slider para activar la votación.</span>';
        return;
    }

    // Construimos las porciones. Caso especial: una sola IA con peso → círculo completo.
    const active = items.filter(it => it.w > 0);
    let svgInner = '';
    if(active.length === 1) {
        svgInner += `<circle cx="${cx}" cy="${cy}" r="${rOuter}" fill="${active[0].color}"/>`;
    } else {
        let cumAngle = -90; // empieza en las 12 en punto
        for(const it of active) {
            const angle = (it.w / total) * 360;
            const a0 = cumAngle * Math.PI / 180;
            const a1 = (cumAngle + angle) * Math.PI / 180;
            const x0 = cx + rOuter * Math.cos(a0);
            const y0 = cy + rOuter * Math.sin(a0);
            const x1 = cx + rOuter * Math.cos(a1);
            const y1 = cy + rOuter * Math.sin(a1);
            const large = angle > 180 ? 1 : 0;
            svgInner += `<path d="M ${cx} ${cy} L ${x0} ${y0} `
                      + `A ${rOuter} ${rOuter} 0 ${large} 1 ${x1} ${y1} Z" `
                      + `fill="${it.color}" stroke="#0a0a0a" stroke-width="1"/>`;
            cumAngle += angle;
        }
    }
    // Agujero central (efecto donut) + total
    svgInner += `<circle cx="${cx}" cy="${cy}" r="${rInner}" fill="#0f0f0f"/>`;
    svgInner += `<text x="${cx}" y="${cy-3}" text-anchor="middle" font-size="8" fill="#888" letter-spacing="0.5">PESO TOTAL</text>`;
    svgInner += `<text x="${cx}" y="${cy+13}" text-anchor="middle" font-size="16" fill="#fff" font-weight="700">${total}</text>`;
    svg.innerHTML = svgInner;

    // Leyenda con porcentajes (sólo las IAs con peso > 0)
    if(legend) {
        const rows = active.map(it => {
            const pct = (it.w / total * 100).toFixed(1);
            return `<span style="display:inline-flex;align-items:center;gap:3px;margin:1px 4px;white-space:nowrap">
                        <span style="width:9px;height:9px;background:${it.color};border-radius:2px;display:inline-block"></span>
                        <span style="color:${it.color};font-weight:600">${it.name}</span>
                        <span style="color:#888">${pct}%</span>
                    </span>`;
        }).join('');
        legend.innerHTML = rows;
    }
}

async function loadConfig() {
    const status = document.getElementById('cfg-status');
    if(status) { status.textContent = '⏳ Cargando config…'; status.style.color = '#ffb066'; }
    console.log('[panel] loadConfig() fetching /api/config…');
    try {
        const r = await fetch('/api/config?key=' + encodeURIComponent(KEY));
        console.log('[panel] /api/config status=' + r.status);
        if(!r.ok) {
            const msg = '❌ No se pudo cargar config (HTTP ' + r.status + ')';
            if(status) { status.textContent = msg; status.style.color = '#dc3545'; }
            console.error('[panel] ' + msg);
            return;
        }
        let c = {};
        try { c = await r.json(); } catch(parseErr) {
            const msg = '❌ Respuesta inválida del relay';
            if(status) { status.textContent = msg; status.style.color = '#dc3545'; }
            console.error('[panel] JSON parse error:', parseErr);
            return;
        }
        console.log('[panel] config keys recibidas:', Object.keys(c).length);
        const set = (id, v) => { const el = document.getElementById(id);
                                  if(el) el.value = v || ''; };
        // Analyzers (text/image)
        set('cfg_ant',             c.anthropic_key);
        set('cfg_ant_bk',          c.anthropic_key_backup);
        set('cfg_ant_model',       c.anthropic_model);
        // Anthropic también participa en OCR-video (mismo key, modelo separado)
        set('cfg_ant_video_model', c.claude_video_ocr_model);
        set('cfg_oai',             c.openai_key);
        set('cfg_oai_bk',          c.openai_key_backup);
        set('cfg_oai_model',       c.openai_model);
        // OpenAI también participa en OCR-video (mismo key, modelo separado)
        set('cfg_oai_video_model', c.openai_video_ocr_model);
        // OpenAI tuning agéntico (Responses API): effort + max_tool_calls + allowed_domains
        const effortEl = document.getElementById('cfg_oai_effort');
        if(effortEl) effortEl.value = (c.openai_reasoning_effort || 'medium');
        const mtcEl = document.getElementById('cfg_oai_max_tools');
        if(mtcEl) mtcEl.value = String(c.openai_max_tool_calls || 4);
        const domsEl = document.getElementById('cfg_oai_domains');
        if(domsEl) {
            const arr = Array.isArray(c.openai_allowed_domains) ? c.openai_allowed_domains : [];
            domsEl.value = arr.join('\n');
        }
        set('cfg_gem',             c.gemini_key);
        set('cfg_gem_bk',          c.gemini_key_backup);
        set('cfg_gem_model',       c.gemini_model);
        set('cfg_gem_video_model', c.gemini_video_model);
        set('cfg_dsk',             c.deepseek_key);
        set('cfg_dsk_bk',          c.deepseek_key_backup);
        set('cfg_dsk_model',       c.deepseek_model);
        set('cfg_mst',             c.mistral_key);
        set('cfg_mst_bk',          c.mistral_key_backup);
        set('cfg_mst_model',       c.mistral_model);
        // OCRs de video
        set('cfg_qwn',             c.qwen_key);
        set('cfg_qwn_bk',          c.qwen_key_backup);
        set('cfg_qwn_model',       c.qwen_video_model);
        set('cfg_kmi',             c.kimi_key);
        set('cfg_kmi_bk',          c.kimi_key_backup);
        set('cfg_kmi_model',       c.kimi_video_model);
        // MiMo: analyzer + OCR video (mismo proveedor)
        set('cfg_mim',             c.mimo_key);
        set('cfg_mim_bk',          c.mimo_key_backup);
        set('cfg_mim_model',       c.mimo_model);
        set('cfg_mim_video_model', c.mimo_video_model);
        // NVIDIA Nemotron (OCR-video puro con su propia key)
        set('cfg_nv',              c.nvidia_key);
        set('cfg_nv_bk',           c.nvidia_key_backup);
        set('cfg_nv_model',        c.nvidia_video_ocr_model);
        // Tavily (paso intermedio OCR → razonamiento)
        set('cfg_tav',             c.tavily_key);
        set('cfg_tav_bk',          c.tavily_key_backup);
        set('cfg_tav_depth',       c.tavily_search_depth || 'basic');
        if(c.tavily_max_results      != null) set('cfg_tav_max',      String(c.tavily_max_results));
        if(c.tavily_http_timeout_s   != null) set('cfg_tav_http_to',  String(c.tavily_http_timeout_s));
        if(c.tavily_total_deadline_s != null) set('cfg_tav_deadline', String(c.tavily_total_deadline_s));
        const tavEn = document.getElementById('cfg_tav_enabled');
        if(tavEn) tavEn.checked = (c.tavily_enabled !== false);
        // Pesos (range 0..10). El label de valor también se sincroniza.
        const setW = (id, v) => {
            const el = document.getElementById(id);
            if(!el) return;
            const n = (v == null) ? 1 : Math.max(0, Math.min(10, parseInt(v) || 0));
            el.value = String(n);
            const vlabel = document.getElementById(id + '_v');
            if(vlabel) vlabel.textContent = String(n);
        };
        setW('cfg_w_ant', c.anthropic_weight);
        setW('cfg_w_oai', c.openai_weight);
        setW('cfg_w_gem', c.gemini_weight);
        setW('cfg_w_dsk', c.deepseek_weight);
        setW('cfg_w_mst', c.mistral_weight);
        setW('cfg_w_mim', c.mimo_weight);
        renderWeightsPie();
        if(status) {
            status.textContent = '✓ Config cargada (' + Object.keys(c).length + ' campos)';
            status.style.color = '#198754';
        }
        console.log('[panel] loadConfig() OK');
    } catch(e) {
        const msg = '❌ ' + (e.message || e);
        if(status) { status.textContent = msg; status.style.color = '#dc3545'; }
        console.error('[panel] loadConfig falló:', e);
    }
}

// ── Auto-guardado de pesos al mover slider ──────────────────────────────────
// Cuando el usuario suelta cualquier slider de peso, esto se dispara automática-
// mente. NO envía API keys ni modelos (esos siguen requiriendo "Guardar todo")
// para evitar pisar inputs que el usuario aún esté editando.
let _autoSaveTimer = null;
function autoSaveWeights() {
    const status = document.getElementById('weights-status');
    if(status) { status.textContent = '⏳ guardando…'; status.style.color = '#ffb066'; }
    clearTimeout(_autoSaveTimer);
    // Debounce 300ms: si el usuario mueve varios sliders rápido, agrupa en 1 POST.
    _autoSaveTimer = setTimeout(async () => {
        const v = id => (document.getElementById(id)?.value || '').trim();
        const body = {
            anthropic_weight: parseInt(v('cfg_w_ant')) || 0,
            openai_weight:    parseInt(v('cfg_w_oai')) || 0,
            gemini_weight:    parseInt(v('cfg_w_gem')) || 0,
            deepseek_weight:  parseInt(v('cfg_w_dsk')) || 0,
            mistral_weight:   parseInt(v('cfg_w_mst')) || 0,
            mimo_weight:      parseInt(v('cfg_w_mim')) || 0,
        };
        try {
            const r = await fetch('/api/config', {
                method:  'POST',
                headers: {'Content-Type':'application/json','X-Api-Key':KEY},
                body:    JSON.stringify(body),
            });
            let data = {};
            try { data = await r.json(); } catch(_) { data = {}; }
            if(!r.ok) throw new Error(data.detail || ('HTTP ' + r.status));
            if(status) {
                if(data.persisted === false) {
                    status.textContent = '⚠ aplicado en RAM (DB caída)';
                    status.style.color = '#ffc107';
                } else {
                    status.textContent = '✓ guardado · ' + new Date().toLocaleTimeString('es');
                    status.style.color = '#198754';
                    // Borra el mensaje a los 3s para no acumular ruido visual.
                    setTimeout(() => {
                        if(status.textContent.startsWith('✓')) status.textContent = '';
                    }, 3000);
                }
            }
            toast('⚖️ Pesos actualizados');
        } catch(e) {
            if(status) { status.textContent = '❌ ' + e.message; status.style.color = '#dc3545'; }
        }
    }, 300);
}

// ── Config: guardar TODO (envía siempre todos los campos, "" = limpiar) ─────
async function saveKeys() {
    const status = document.getElementById('cfg-status');
    const v = id => (document.getElementById(id)?.value || '').trim();
    const body = {
        // Analyzers
        anthropic_key:        v('cfg_ant'),
        anthropic_key_backup: v('cfg_ant_bk'),
        openai_key:           v('cfg_oai'),
        openai_key_backup:    v('cfg_oai_bk'),
        gemini_key:           v('cfg_gem'),
        gemini_key_backup:    v('cfg_gem_bk'),
        deepseek_key:         v('cfg_dsk'),
        deepseek_key_backup:  v('cfg_dsk_bk'),
        mistral_key:          v('cfg_mst'),
        mistral_key_backup:   v('cfg_mst_bk'),
        // OCRs video
        qwen_key:             v('cfg_qwn'),
        qwen_key_backup:      v('cfg_qwn_bk'),
        kimi_key:             v('cfg_kmi'),
        kimi_key_backup:      v('cfg_kmi_bk'),
        // MiMo (analyzer + OCR video, full-modal Xiaomi)
        mimo_key:             v('cfg_mim'),
        mimo_key_backup:      v('cfg_mim_bk'),
        // NVIDIA Nemotron OCR (key propia)
        nvidia_key:           v('cfg_nv'),
        nvidia_key_backup:    v('cfg_nv_bk'),
        // Tavily (paso intermedio OCR → razonamiento)
        tavily_key:           v('cfg_tav'),
        tavily_key_backup:    v('cfg_tav_bk'),
        tavily_enabled:       !!document.getElementById('cfg_tav_enabled')?.checked,
        tavily_search_depth:  v('cfg_tav_depth') || 'basic',
        tavily_max_results:   parseInt(v('cfg_tav_max'))      || 3,
        tavily_http_timeout_s:   parseFloat(v('cfg_tav_http_to'))  || 6.0,
        tavily_total_deadline_s: parseFloat(v('cfg_tav_deadline')) || 8.0,
        // Modelos
        anthropic_model:      v('cfg_ant_model'),
        openai_model:         v('cfg_oai_model'),
        gemini_model:         v('cfg_gem_model'),
        gemini_video_model:   v('cfg_gem_video_model'),
        deepseek_model:       v('cfg_dsk_model'),
        mistral_model:        v('cfg_mst_model'),
        qwen_video_model:     v('cfg_qwn_model'),
        kimi_video_model:     v('cfg_kmi_model'),
        mimo_model:           v('cfg_mim_model'),
        mimo_video_model:     v('cfg_mim_video_model'),
        // Claude/OpenAI OCR-video: misma key que analyzer, modelo separado
        claude_video_ocr_model: v('cfg_ant_video_model'),
        openai_video_ocr_model: v('cfg_oai_video_model'),
        nvidia_video_ocr_model: v('cfg_nv_model'),
        // OpenAI tuning agéntico (Responses API · web_search GA)
        openai_reasoning_effort: v('cfg_oai_effort'),
        openai_max_tool_calls:   parseInt(v('cfg_oai_max_tools')) || 4,
        openai_allowed_domains:  v('cfg_oai_domains')
                                    .split(/[\n,]/)
                                    .map(s => s.trim())
                                    .filter(Boolean)
                                    .slice(0, 100),
        // Pesos en la votación de fusionar (0..10, int)
        anthropic_weight:     parseInt(v('cfg_w_ant')) || 0,
        openai_weight:        parseInt(v('cfg_w_oai')) || 0,
        gemini_weight:        parseInt(v('cfg_w_gem')) || 0,
        deepseek_weight:      parseInt(v('cfg_w_dsk')) || 0,
        mistral_weight:       parseInt(v('cfg_w_mst')) || 0,
        mimo_weight:          parseInt(v('cfg_w_mim')) || 0,
    };
    try {
        const r = await fetch('/api/config', {
            method:  'POST',
            headers: {'Content-Type':'application/json','X-Api-Key':KEY},
            body:    JSON.stringify(body)
        });
        // Parsing defensivo: si el server devuelve un 500 con body no-JSON
        // (proxy error, etc.), JSON.parse falla con "Unexpected token". Caja-fuerte.
        let data = {};
        try { data = await r.json(); } catch(_) { data = {}; }
        if(!r.ok) throw new Error(data.detail || ('HTTP ' + r.status));
        const n = (data.changes || []).length;
        if(data.persisted === false) {
            // Caso especial: la config se aplicó en RAM pero ninguna DB la aceptó.
            // Server responde 503 → ya no entramos aquí (lanza throw arriba).
            // Pero por defensiva, también lo cubrimos por si el flujo cambia.
            if(status) status.textContent = '⚠️ Aplicado en RAM pero NO persistido (' + n + ' cambios)';
            toast('⚠️ Cambios aplicados temporalmente — DBs no disponibles');
        } else {
            if(status) status.textContent = '✅ Guardado: ' + n + ' cambios';
            toast('✅ Config actualizada');
        }
    } catch(e) {
        if(status) status.textContent = '❌ ' + e.message;
        toast('❌ ' + e.message);
    }
}

// ── Modal de errores ─────────────────────────────────────────────────────────
async function showErrors() {
    const body = document.getElementById('errors-body');
    const modal = document.getElementById('errors-modal');
    modal.style.display = '';
    body.textContent = 'Cargando...';
    try {
        const r = await fetch('/api/errors?key=' + encodeURIComponent(KEY) + '&limit=100');
        if(!r.ok) throw new Error('HTTP ' + r.status);
        const data = await r.json();
        if(data.errors.length === 0) { body.textContent = '✓ Sin errores recientes'; return; }
        body.textContent = data.errors.map(e => {
            const t = new Date(e.t * 1000).toLocaleString('es');
            // Si el mismo error se repitió (dedup), mostramos cuántas veces y la última.
            const n = e.count || 1;
            if(n > 1) {
                const lastT = e.last_t ? new Date(e.last_t * 1000).toLocaleString('es') : t;
                return `[${t} → ${lastT}] [${e.level}] [${e.thread}] (x${n})\n${e.msg}`;
            }
            return `[${t}] [${e.level}] [${e.thread}]\n${e.msg}`;
        }).join('\n\n');
    } catch(e) { body.textContent = '❌ ' + e.message; }
}

async function clearErrors() {
    if(!confirm('¿Borrar el log de errores?')) return;
    try {
        const r = await fetch('/api/errors', {
            method: 'DELETE',
            headers: {'X-Api-Key': KEY}
        });
        if(!r.ok) throw new Error('HTTP ' + r.status);
        document.getElementById('errors-body').textContent = '✓ Borrado';
        toast('🗑 Errores limpiados');
        pollErrorCount();
    } catch(e) { toast('❌ ' + e.message); }
}

async function pollErrorCount() {
    try {
        const r = await fetch('/health');
        if(!r.ok) return;
        const h = await r.json();
        const n = (h.errors && h.errors.in_log) || 0;
        const el = document.getElementById('stat-errlog');
        if(el) {
            el.textContent = '📋 ' + n + ' logs';
            el.style.background = n > 0 ? '#2a0a0a' : '';
        }
    } catch(e) {}
}

setInterval(pollErrorCount, 15000); pollErrorCount();

function toggleEye(id) {
    const el = document.getElementById(id);
    if(!el) return;
    el.type = el.type === 'password' ? 'text' : 'password';
}

// ── Exportar config a fichero local ──────────────────────────────────────────
async function exportConfig() {
    const status = document.getElementById('cfg-status');
    try {
        const url = '/api/config/export?key=' + encodeURIComponent(KEY);
        const r = await fetch(url);
        if(!r.ok) throw new Error('HTTP ' + r.status);
        const blob = await r.blob();
        const cd   = r.headers.get('Content-Disposition') || '';
        const fnMatch = /filename="([^"]+)"/.exec(cd);
        const fname = fnMatch ? fnMatch[1] : 'relay_config.json';
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = fname;
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
        if(status) status.textContent = '✓ Descargado ' + fname;
        toast('📥 Backup descargado');
    } catch(e) {
        if(status) status.textContent = '❌ Export: ' + e.message;
        toast('❌ ' + e.message);
    }
}

// ── Importar config desde fichero ─────────────────────────────────────────────
async function importConfig(input) {
    const status = document.getElementById('cfg-status');
    const file = input.files && input.files[0];
    if(!file) return;
    try {
        const text = await file.text();
        const data = JSON.parse(text);
        const r = await fetch('/api/config/import', {
            method:  'POST',
            headers: {'Content-Type': 'application/json', 'X-Api-Key': KEY},
            body:    JSON.stringify(data),
        });
        if(!r.ok) throw new Error((await r.json()).detail || r.status);
        const res = await r.json();
        if(status) status.textContent = '✓ Importado ' + res.applied.length + ' claves';
        toast('📤 Importado: ' + res.applied.length + ' claves');
        await loadConfig();   // refrescar inputs con los nuevos valores
    } catch(e) {
        if(status) status.textContent = '❌ Import: ' + e.message;
        toast('❌ ' + e.message);
    } finally {
        input.value = '';     // permitir re-importar el mismo archivo
    }
}

// Auto-cargar config en TODOS los escenarios:
//  1) Cuando el usuario abre el <details> (toggle event).
//  2) Inmediatamente al cargar la página, así los campos están poblados incluso
//     antes de abrir el panel (evita "campos vacíos" si el usuario abre rápido).
//  3) Si el panel ya está abierto en el momento del page-load (browser state
//     restore, navegación back/forward), el toggle no dispara → cargamos aquí.
//
// _loaded actúa como guard contra dobles cargas — la primera que llegue gana.
(function autoLoadConfig() {
    const panel = document.getElementById('cfg-panel');
    const tryLoad = () => {
        if(panel && panel._loaded) return;
        if(panel) panel._loaded = true;
        loadConfig();
    };
    if(panel) {
        panel.addEventListener('toggle', function() { if(this.open) tryLoad(); });
        if(panel.open) tryLoad();
    }
    // Failsafe: si por alguna razón el panel aún no existe (timing), cargar igualmente.
    if(document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', tryLoad, { once: true });
    } else {
        tryLoad();
    }
})();

// ── Toast ─────────────────────────────────────────────────────────────────────
function toast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg; t.style.display = 'block';
    setTimeout(() => t.style.display = 'none', 2200);
}

// ── Atajos teclado (cuando hay 1 sola tarjeta activa) ────────────────────────
let _ki = 0;
document.addEventListener('keydown', ev => {
    const cards = document.querySelectorAll('.card[data-status="awaiting_review"]');
    if(cards.length !== 1) return;
    const jid = cards[0].dataset.jid;
    if(ev.key === 'Enter') { ev.preventDefault(); sendToMobile(jid); return; }
    if(/^[1-9]$/.test(ev.key)) {
        _ki = parseInt(ev.key, 10) - 1; ev.preventDefault();
        const keys = document.querySelectorAll(`#keys-${jid} .key`);
        keys.forEach((k,i) => k.style.outline = i===_ki ? '3px solid #9b85ff' : '');
        return;
    }
    const k = ev.key.toUpperCase();
    if(_LET.includes(k)) {
        const keys = document.querySelectorAll(`#keys-${jid} .key`);
        if(keys[_ki]) {
            const lEl = keys[_ki].querySelector('.key-letter');
            lEl.textContent = k;
            keys[_ki].className = keys[_ki].className.replace(/k-[ABCDX]/, 'k-' + k);
            _markDirty(jid); _syncTxt(jid); _updPreview(jid);
            _ki = Math.min(_ki + 1, keys.length - 1);
            keys.forEach((kk,i) => kk.style.outline = i===_ki ? '3px solid #9b85ff' : '');
        }
        ev.preventDefault();
    }
});

// Polling adaptativo: chain de setTimeout en vez de setInterval fijo.
// - Si hay jobs activos (pending/awaiting_review) → 1 s (latencia rápida).
// - Si todo está done/error/idle → 5 s (reduce carga del servidor 5×).
// _activeJobsCount lo actualiza applyJobs() después de cada poll exitoso.
// Si el poll falla, mantiene el intervalo activo para reintentar agresivo.
let _activeJobsCount = 0;
const POLL_FAST_MS = 1000;
const POLL_IDLE_MS = 5000;
function scheduleNextPoll() {
    const next = (_activeJobsCount > 0 || _pollFails > 0) ? POLL_FAST_MS : POLL_IDLE_MS;
    setTimeout(async () => {
        try { await poll(); } catch(_) {}
        scheduleNextPoll();
    }, next);
}
setInterval(ticks, 1000);
poll().finally(scheduleNextPoll); ticks();
