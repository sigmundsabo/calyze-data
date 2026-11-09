# Runbook — mit csinálj, ha valami elromlik

Ez a lista lépésről lépésre követhető programozói tudás nélkül. A parancsokat
külön blokkokban adjuk meg; mindegyiket a saját Terminálodban futtasd.

## 1. A napi futás elhasalt (e-mail a GitHubtól)

1. Nyisd meg a futást: <https://github.com/sigmundsabo/calyze-data/actions>
2. Nézd meg, melyik lépés piros:
   - **Letöltés:** adatforrás-hiba. Nézd meg a `provider_*` sorokat: ha
     `provider_blocked`, a forrás kitiltott minket; ha `missing`, néhány papír
     kimaradt. Egy nap kimaradása nem hiba, csak jelölve lesz.
   - **Feature-ök:** jellemzően hiányzó FRED-kulcs vagy elérhetetlen FRED.
   - **Napi becslés:** ha „Nincs betanított modell”, futtasd a tanítást (lásd 3.).
   - **A napi lenyomat commitolása:** ha a push ütközött, indítsd újra a futást.
3. Újraindítás:

```bash
gh workflow run ingest.yml -R sigmundsabo/calyze-data
```

**Fontos:** a becslés egy napra csak egyszer készül. Ha a csomag már
lementődött, az újrafuttatás nem írja felül — ez szándékos.

## 2. Egy nap kimaradt

Nem pótoljuk. A kiesett nap kiesett napként látszik, és a felület is így írja
ki. Visszamenőleg becslést gyártani tilos.

## 3. A modell újratanítása

Hetente magától fut (vasárnap 08:00 UTC). Kézzel:

```bash
gh workflow run train.yml -R sigmundsabo/calyze-data -f task=train
```

## 4. A mérés újraszámolása (backtest)

Órás nagyságrend, a privát tárba ír:

```bash
gh workflow run train.yml -R sigmundsabo/calyze-data -f task=backtest
```

A tárolt mérés gyors kiírása (nem számol újra):

```bash
gh workflow run train.yml -R sigmundsabo/calyze-data -f task=report
```

## 5. „Halott ember kapcsoló” riasztás (a futás el sem indult)

1. Ellenőrizd, hogy a GitHub Actions nem áll-e: <https://www.githubstatus.com/>
2. Nézd meg, nincs-e letiltva az ütemezés (hosszú repó-inaktivitás után a
   GitHub letiltja; a napi lenyomat-commit ezt kizárja).
3. Indítsd el kézzel (lásd 1.3).

## 6. Kulcs szivárgott vagy gyanús a naplóban valami

1. Vond vissza és cseréld a kulcsot a szolgáltatónál.
2. Tedd be az újat: `gh secret set <NÉV> -R sigmundsabo/calyze-data`
3. Futtasd újra a napi futást, és nézd meg, hogy lefut-e.
4. Nézd át, mikor óta volt kint a kulcs (a futások naplója nyilvános).

## 7. Betelt a tár vagy a keret

- Supabase tár: a `prices_daily` évfájljai a legnagyobbak. Régi évek
  tömöríthetők vagy archiválhatók.
- A feature-öket nem tároljuk, azok újraszámolódnak.
