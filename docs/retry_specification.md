# Combat Retry Specification / 戦闘リトライ仕様

## 1. Boss or Elite / ボスまたはエリートの場合

- **Retry Limit:** Initial 10 attempts. / **リトライ回数:** 最初は10回。
- **Winning Condition for Progression:** / **勝利時の進捗条件:**
    - If `HP loss <= Floor number`, proceed to the next floor. / `HPの減少 <= フロア階数` ならば、次のフロアへ。
    - If `HP loss > Floor number`, retry the battle. / `HPの減少 > フロア階数` ならば、リトライ。
- **Failure to win within 10 attempts:** / **10回までに一度も勝利できなかった場合:**
    - Backtrack to previous save data. / セーブデータを遡る（バックトラック）。
- **Winning within 10 attempts but not meeting HP threshold:** / **10回までに勝利はできたが、HP条件を満たせなかった場合:**
    - Increase Retry Limit to 20 attempts. / リトライ回数を20回に増やす。
    - **New Progression Condition:** `HP loss <= Median of previous wins`. / **新しい進捗条件:** `HPの減少 <= これまでの勝利時の中央値`。
    - If still not progressed by 20th attempt, backtrack to previous save data. / 20回までに進捗条件を満たせなかった場合、セーブデータを遡る。

## 2. Normal Enemy / 通常敵の場合

- **Retry Limit:** Max 3 attempts. / **リトライ回数:** 最大3回。
- **Winning Condition for Progression:** / **勝利時の進捗条件:**
    - If `HP loss <= Floor number`, proceed to the next floor. / `HPの減少 <= フロア階数` ならば、次のフロアへ。
    - If `HP loss > Floor number`, retry the battle. / `HPの減少 > フロア階数` ならば、リトライ。
- **Failure to progress within 3 attempts:** / **3回までに次のフロアへ進めなかった場合:**
    - Backtrack to previous save data. / セーブデータを遡る（バックトラック）。
