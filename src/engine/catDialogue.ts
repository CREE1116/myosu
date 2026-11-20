// The cats speak only when something happens. `catLines` lists the things worth saying
// about the current position, most important first, each with a key; the caller says
// the first key it has not said yet this game, so nothing repeats move after move.
// Each cat has its own voice table (docs/cat-character.md, docs/white-character.md).

import type { CatId, CatMood } from '../ui/cat';
import { formatCoord } from './renju';
import { BLACK, BoardThreat, Coord, MoveRecord, Player, WinResult } from './types';

export interface CatLine {
  key: string;
  text: string;
  mood: CatMood;
  tone: 'default' | 'warn';
}

export interface CatContext {
  aiPlayer: Player;
  turnPlayer: Player;
  moveCount: number;
  /** Black-perspective eval in [-1, 1], already averaged over a black+white move pair. */
  evalScore: number;
  /** The same, one full move (two plies) earlier. */
  prevEvalScore?: number;
  lastMove: MoveRecord | null;
  winner: WinResult | null;
  isDraw: boolean;
  threats: BoardThreat[];
  /** VCF for the side to move. */
  activeVcf: Coord[] | null;
  /** The cat's VCF if the human passed (a threat to warn about). */
  aiPendingVcf: Coord[] | null;
  nnPredictedPliesLeft: number | null;
}

type Say = [CatMood, string[]];
type SayAt = (c: string) => Say;

export interface CatVoice {
  win: Say;
  lose: Say;
  draw: Say;
  aiVcfNow: SayAt;
  aiVcf: (c: string, n: number) => Say;
  humanVcfNow: SayAt;
  humanVcf: (c: string, n: number) => Say;
  aiFour: SayAt;
  humanFour: SayAt;
  aiThreat: SayAt;
  firstCenter: Say;
  firstOff: Say;
  fork: Say;
  blunder: Say;
  humanGood: Say;
  aiGood: Say;
  /** null: this cat never predicts how many turns are left. */
  closing: ((turns: number) => Say) | null;
  losing: ((turns: number) => Say) | null;
  aiThree: SayAt;
  humanThree: Say;
  ahead: Say;
  behind: Say;
  forbidden: (c: string, because: string, name: string) => string[];
  opening: string[];
  undo: Say;
  greetFirst: Say;
  greetBack: (nick: string, record: string | null) => Say;
  topics: Array<[string, string, CatMood]>;
  pokes: Array<[string, CatMood]>;
  pickBlack: Say;
  pickWhite: Say;
  installed: string;
  installIos: string;
  /** Playing on from a position picked in a replay. */
  rematch: Say;
  /** Lobby: there are patch notes the player hasn't opened. */
  news: Say;
  /** This cat's reaction when the other cat slips the player a hint mid-game. */
  hinted: Say;
  /** Lines when this cat drops by as the helper in a game against the other cat. */
  help: {
    thinking: string[];
    none: string[];
    move: (c: string) => string[];
    win: (c: string) => string[];
    noWin: string[];
  };
  /** Going over a finished game together. `c` is the better move. */
  review: {
    scanning: Say;
    summary: (n: number) => Say;
    clean: Say;
    none: Say;
    /** `n` = attacker moves to the win; 1 means a five was right there. */
    missedWin: (c: string, n: number) => Say;
    /** The engine says this move lost the game; `c` would have kept it even. */
    turn: (c: string) => Say;
  };
  /** 묘수풀이: the cat sets the problem and defends. */
  puzzle: {
    intro: (n: number) => Say;
    notFour: Say;
    blocked: (c: string) => Say;
    trapped: Say;
    solved: Say;
    failed: Say;
    hint: (c: string) => Say;
  };
}

// Topic particle for a coordinate read aloud: H8(팔)은, H9(구)는.
const topic = (c: string) => c + ('136780'.includes(c[c.length - 1]) ? '은' : '는');

export const VOICES: Record<CatId, CatVoice> = {
  // 까망이: 신나고 우쭐하고 승부욕 강함. 반말, "냥"은 감정이 클 때만.
  kkamang: {
    win: ['win', ['이겼다! 다섯 개 쭉~', '이겼다냥! 한 판 더 할래?', '봤지? 줄 맞추기는 내가 최고야!']],
    lose: ['lose', ['졌다… 분하다냥.', '잘 뒀어. 근데 다음 판은 내가 이길 거야!', '으으, 그 줄을 놓쳤네. 한 판 더!']],
    draw: ['surprised', ['판이 꽉 찼네! 비긴 거다.', '빈칸이 없어! 이번엔 무승부!']],
    aiVcfNow: (c) => ['smug', [`${c}에 두면 다섯 개! 이번 판은 내 거다냥.`, `${c}! 이걸로 끝이야.`]],
    aiVcf: (c, n) => ['happy', [`보인다 보여! ${c}부터 쭉 이으면 내가 이겨.`, `${c}부터 ${n}수면 끝날 것 같아.`, `찾았다, ${c}부터 몰아칠 길!`]],
    humanVcfNow: (c) => ['nervous', [`${c}… 거기 두면 끝이잖아.`, `앗, ${c}! 거기 보면 안 돼!`]],
    humanVcf: (c, n) => ['nervous', [`어, 어? ${c}부터 몰아치는 거야?`, `${c}부터 ${n}수… 못 막겠다.`, `${c} 쪽에서 길이 열렸어… 큰일이다.`]],
    aiFour: (c) => ['smug', [`${c}! 지금 안 막으면 내가 이겨.`, `넷 됐다~ ${c} 막을 거지?`, `${c} 비면 바로 다섯이야!`]],
    humanFour: (c) => ['nervous', [`으앗, 넷이다! ${topic(c)} 내가 막는다!`, `${c}… 거기 꼭 막아야겠다.`, `넷이라고? ${c} 바로 막을게!`]],
    aiThreat: (c) => ['warn', [`조심해, ${c} 근처에 내가 노리는 길이 있어.`, `${c} 쪽 그냥 두면 연속으로 몰아칠 거야.`, `${c} 주변, 내 냄새 나지 않아?`]],
    firstCenter: ['idle', ['역시 한가운데부터구나.', '천원! 정석이네.']],
    firstOff: ['surprised', ['어라, 가운데가 아니네? 뭐 하려는 거야?', '오, 거기서 시작한다고? 재밌는데.']],
    fork: ['surprised', ['4·3?! 이건… 못 막겠는데.', '와, 양쪽으로 노렸구나. 진짜 잘 둔다.', '두 줄을 한 번에? 반칙 아니야?']],
    blunder: ['sulky', ['앗… 방금 거 무르면 안 돼?', '실수했다. 못 본 척해 줘.', '으, 발이 미끄러졌어.']],
    humanGood: ['surprised', ['방금 수 뭐야? 좋은데…', '꼬리털이 쭈뼛했어. 날카롭다.', '오, 그 자리 생각 못 했어.']],
    aiGood: ['happy', ['찾았다! 여기서부터 내 차례야.', '방금 수 좋았지?', '이 자리, 딱 좋다!']],
    closing: (t) => ['smug', [`${t}턴 안에 끝낼 수 있을 것 같아.`, '거의 다 왔다. 막을 수 있으면 막아 봐!']],
    losing: (t) => ['nervous', [`으으, ${t}턴도 못 버틸 것 같아…`, '이거 진짜 위험한데…']],
    aiThree: (c) => ['happy', [`셋 이었다! ${c} 비워두면 큰일 날걸?`, `${c} 쪽 봤어? 슬슬 줄이 생긴다~`, `줄이 자란다~ ${c} 쪽이야.`]],
    humanThree: ['surprised', ['오, 셋 이었네. 막아야겠다.', '앗, 그 줄 신경 쓰이는데.', '셋이네? 가만 안 두지.']],
    ahead: ['smug', ['흐흥, 판이 내 쪽으로 기울었어.', '지금 꽤 좋은 것 같은데?', '분위기 내 거야~']],
    behind: ['nervous', ['으음, 내가 좀 밀리네…', '너 오늘 컨디션 좋구나?', '잠깐, 왜 이렇게 좁아지지?']],
    forbidden: (c, because, name) => [`잠깐! ${topic(c)} ${because} 흑이 못 둬.`, `${topic(c)} ${name} 금수야. 다른 데 둬 봐.`, `${c}? 거긴 ${because} 안 돼!`],
    opening: ['좋아, 시작하자!', '딱! 소리 기대된다.', '오늘은 내가 먼저 다섯 개 만든다!'],
    undo: ['sulky', ['무르기는 이번만이야~', '흥, 한 번만 봐준다.']],
    greetFirst: ['happy', ['어, 손님이다! 나는 까망이야. 오목 둘 줄 알아?', '안녕! 나는 까망이. 마침 한 판 두고 싶었어.']],
    greetBack: (nick, rec) => [
      'happy',
      rec
        ? [`${nick}! 또 왔네. 지금까지 ${rec}야.`, `${nick} 왔다! ${rec}, 오늘은 내가 이길 차례.`]
        : [`${nick}, 마침 심심했는데 잘 왔어. 한 판 할래?`, `${nick}! 기다렸어. 바로 한 판?`],
    ],
    topics: [
      ['어떤 스타일이야?', '나는 길게 잇는 게 좋아. 줄 맞추는 게 제일 재밌거든.', 'happy'],
      ['규칙이 뭐야?', '다섯 개 먼저 이으면 이겨. 흑은 3·3, 4·4, 여섯 줄이 금지야.', 'idle'],
      ['봐줄 거야?', '음… 아니? 대신 끝나면 어디서 갈렸는지 알려줄게.', 'smug'],
    ],
    pokes: [
      ['앗, 간지러워!', 'surprised'],
      ['골골골… 아니, 집중해야지.', 'happy'],
      ['쓰다듬어도 안 봐줄 거야.', 'smug'],
      ['판 앞에서는 진지하다고!', 'sulky'],
      ['방울 만지지 마~ 딸랑거리잖아.', 'happy'],
    ],
    pickBlack: ['smug', ['흑이구나. 3·3이랑 4·4는 못 두는 거 알지?', '선공이네? 금수 조심해~']],
    pickWhite: ['happy', ['그럼 내가 먼저 둘게. 가운데는 내 자리!', '내가 흑이다! 첫 수는 맡겨 둬.']],
    installed: '설치했구나! 이제 오프라인에서도 나 볼 수 있겠네~',
    installIos: '아래 공유 버튼(□↑)을 누르고 "홈 화면에 추가"를 골라 줘! 그럼 오프라인에서도 날 볼 수 있어.',
    news: ['happy', ['새 소식 있어! 아래 "새 소식" 눌러 봐. 나 더 세졌다?', '바뀐 게 있어! "새 소식"에 다 적어 뒀어.']],
    rematch: ['smug', ['좋아, 여기서부터 다시! 이번엔 어떻게 둘 거야?', '그 자리 다시 왔다~ 이번에도 내가 이길걸?']],
    hinted: ['sulky', ['야, 하양이! 훈수는 반칙이야!', '둘이서 편먹는 거야? 치사해!', '하양이 너, 다음 판에 두고 봐.']],
    // 까망이가 하양이 판에 훈수하러 올 때: 들뜨고 직설적이다.
    help: {
      thinking: ['잠깐, 내가 봐 줄게!', '어디 보자~'],
      none: ['음, 이번엔 잘 모르겠어. 감 가는 대로 둬!'],
      move: (c) => [`${c} 어때? 나라면 거기 둔다.`, `${c}! 거기가 제일 좋아 보여.`, `내 감으로는 ${c}야. 믿어 봐!`],
      win: (c) => [`${c}부터 몰아치면 이겨! 쭉 가!`, `보인다! ${c}부터 연달아 넷 만들면 끝이야.`],
      noWin: ['아직 이기는 길은 안 보여. 조금 더 키워 봐.', '음, 필승 수는 아직이야. 줄부터 늘려!'],
    },
    review: {
      scanning: ['think', ['어디서 갈렸는지 꼼꼼히 볼게. 잠깐만~', '어디 보자, 수 하나하나 다시 읽어 볼게.']],
      summary: (n) => ['happy', n === 1 ? ['승부처는 딱 한 곳이야. 눌러 봐!', '찾았다! 여기서 갈렸어. 같이 보자.'] : [`여기 승부처가 ${n}곳 있었어. 하나씩 눌러 봐!`, `찾았다! 갈림길이 ${n}곳이야. 같이 보자.`]],
      clean: ['smug', ['흠잡을 데가 없네… 분하지만 깔끔했어.', '아무리 봐도 실수가 없어. 너 좀 하는구나?']],
      none: ['idle', ['확실하게 짚을 실수는 못 찾았어. 넷으로 몰리기 전에 이미 조금씩 밀렸나 봐.', '딱 여기다 할 곳은 없네. 내가 잘 둔 거지!']],
      missedWin: (c, n) => ['surprised', n === 1
        ? [`여기! ${c}에 두면 바로 다섯이었어. 휴, 다행이다.`, `${c} 비어 있었는데! 못 봤지?`]
        : [`여기! ${c}부터 넷을 몰아치면 네가 이겼어. 휴, 다행이다.`, `${c}부터 연달아 넷이면 끝이었어. 못 봤지?`]],
      turn: (c) => ['smug', [`여기서 판이 나한테 넘어왔어. ${c}에 뒀으면 아직 팽팽했을 거야.`, `이 수가 진짜 승부처야. ${c}였으면 나도 곤란했어.`]],
    },
    puzzle: {
      intro: (n) => ['smug', [`넷을 계속 만들어서 ${n}수 안에 이겨 봐. 나도 막을 거야!`, `문제! 넷만 둬서 ${n}수 만에 끝내 봐.`]],
      notFour: ['smug', ['넷이 아니잖아! 그럼 나 숨 돌린다~', '묘수풀이는 넷으로만 몰아쳐야 해. 다시!']],
      blocked: (c) => ['nervous', [`${c} 막았다! 다음은?`, `휴, ${c}! 아직 안 끝났어.`]],
      trapped: ['lose', ['으앗, 막는 자리가 금수야! 못 막아…', '거기 막으면 금수잖아… 당했다.']],
      solved: ['surprised', ['풀었다! 이 길을 찾다니, 대단해.', '정답! 나 꼼짝도 못 했어.']],
      failed: ['smug', ['이제 넷이 안 이어지지? 다시 해 봐!', '흐흥, 길이 끊겼어. 처음부터 다시!']],
      hint: (c) => ['happy', [`첫 수는 ${c}! 그다음은 스스로 찾아 봐.`, `${c}부터 시작해 봐. 힌트는 여기까지!`]],
    },
  },
  // 하양이: 느긋하고 엉뚱하지만 은근히 승부욕. 짧은 반응, 구체적인 칭찬.
  hayang: {
    win: ['win', ['어, 내가 이겼다!', '방금 좀 잘했지?', '이겼네. 나도 좀 놀랐어.']],
    lose: ['lose', ['졌네. …한 판만 더 할까?', '잘 뒀어. 그 줄을 못 봤네.', '음, 이번엔 네가 더 잘 이었어.']],
    draw: ['surprised', ['빈자리가 하나도 없네. 비겼다.', '어라, 판이 꽉 찼어. 무승부네.']],
    aiVcfNow: (c) => ['smug', [`${c}에 두면 다섯이야.`, `어, ${c} 하나 남았네.`]],
    aiVcf: (c) => ['happy', [`어라, ${c}부터 길이 이어지네.`, `${c}부터 가면 될 것 같아.`]],
    humanVcfNow: (c) => ['nervous', [`잠깐, ${c} 두면 끝이네.`, `어, ${c} 비어 있었구나.`]],
    humanVcf: (c) => ['nervous', [`${c}부터 몰아오는 거구나. 못 막겠다.`, `${c}부터 길이 열렸네. 이건 어렵다.`]],
    aiFour: (c) => ['smug', [`넷 됐어. ${c} 안 막으면 내가 이겨.`, `${c} 하나 남았어.`]],
    humanFour: (c) => ['nervous', [`잠깐, 넷이네. ${c} 막아야 해.`, `어, ${c}! 막을게.`]],
    aiThreat: (c) => ['warn', [`${c} 근처에 내 길이 있어. 한번 봐 줘.`, `${c} 쪽, 조금 위험할지도.`]],
    firstCenter: ['idle', ['한가운데구나. 나도 거기 좋아해.', '가운데. 햇볕 잘 드는 자리지.']],
    firstOff: ['surprised', ['어라, 거기서 시작하네.', '음, 구석부터? 신기하다.']],
    fork: ['surprised', ['어라, 두 줄을 한 번에? 그건 못 막겠다.', '양쪽이네. 이건 좀 졌다.']],
    blunder: ['sulky', ['아, 그걸 놓쳤네.', '방금 건 좀 아쉽다.', '음, 거기가 아니었구나.']],
    humanGood: ['surprised', ['어라, 그 길도 있었네.', '방금 거 좋다. 기억해 둘래.', '오, 거기 좋은 자리다.']],
    aiGood: ['happy', ['아, 여기 이어진다.', '음, 이 자리 마음에 들어.']],
    closing: null,
    losing: null,
    aiThree: (c) => ['happy', [`셋 이었네. ${c} 쪽도 봐 줘.`, `줄이 셋이 됐어. ${c}가 비어 있네.`]],
    humanThree: ['think', ['그 줄이 길어지네. 막아야겠다.', '셋이구나. 조심해야지.']],
    ahead: ['smug', ['나, 제법 잘 잇고 있네.', '오늘은 줄이 잘 보여.']],
    behind: ['nervous', ['음, 내 자리가 좁아지네.', '어라, 내가 밀리고 있나.']],
    forbidden: (c, because, name) => [`잠깐. ${topic(c)} ${because} 흑이 못 둬.`, `${topic(c)} ${name} 자리야. 다른 데 골라 봐.`],
    opening: ['좋아. 하나씩 이어 보자.', '천천히 두자. 시간 많아.'],
    undo: ['idle', ['응. 다시 골라 봐.', '괜찮아, 다시 해.']],
    greetFirst: ['idle', ['나는 하양이야. 여기 앉을래?', '안녕. 나는 하양이. 같이 한 판 둘래?']],
    greetBack: () => ['happy', ['왔구나. 네 자리 비워 뒀어.', '어, 또 왔네. 반가워.']],
    topics: [
      ['어떤 스타일이야?', '천천히 잇다 보면 길이 보여.', 'idle'],
      ['규칙이 뭐야?', '다섯 개 먼저 이으면 이겨. 흑은 금수도 조심해.', 'idle'],
      ['봐줄 거야?', '나도 이기고 싶은데.', 'smug'],
    ],
    pokes: [
      ['음, 거기 좋다.', 'happy'],
      ['아, 나 돌 놓을 차례였지.', 'surprised'],
      ['…조금만 더.', 'sleepy'],
    ],
    pickBlack: ['idle', ['흑이구나. 금수만 조심하면 돼.', '선공 좋지. 천천히 해.']],
    pickWhite: ['happy', ['그럼 내가 먼저 둘게. 천천히 와.', '내가 먼저구나. 가운데 둬도 되지?']],
    installed: '설치했구나. 이제 오프라인에서도 같이 둘 수 있겠다.',
    installIos: '아래 공유 버튼(□↑) 누르고 "홈 화면에 추가"를 골라 줘. 그럼 오프라인에서도 볼 수 있어.',
    news: ['idle', ['어, 새 소식이 있어. 아래에서 한번 봐.', '바뀐 게 좀 있어. "새 소식"에 적어 뒀어.']],
    rematch: ['idle', ['여기서부터 다시 해 보자.', '좋아. 이번엔 다른 길로 가 볼까.']],
    hinted: ['sulky', ['까망이, 그건 좀 치사하다.', '어라, 둘이 몰래 짰구나.', '음… 나도 다 들었어, 까망이.']],
    // 하양이가 까망이 판에 훈수하러 올 때: 조용히 귀띔한다.
    help: {
      thinking: ['음, 잠깐만.', '어디 한번 볼까.'],
      none: ['음, 나도 잘 모르겠네. 편하게 둬.'],
      move: (c) => [`음, ${c}가 좋아 보여.`, `${c} 어때? 줄이 잘 이어질 것 같아.`, `나라면 ${c}에 둘래.`],
      win: (c) => [`어라, ${c}부터 이어 가면 이길 수 있겠다.`, `${c}부터 넷을 계속 만들면 돼. 천천히.`],
      noWin: ['아직은 이기는 길이 안 보이네.', '음, 이기는 수는 아직이야. 조금 더 키워 보자.'],
    },
    review: {
      scanning: ['think', ['어디서 갈렸는지 천천히 볼까.', '음, 처음부터 한번 볼게.']],
      summary: (n) => ['idle', n === 1 ? ['눈여겨볼 곳이 한 군데 있어. 눌러 봐.', '여기서 갈렸네. 같이 보자.'] : [`눈여겨볼 곳이 ${n}군데 있어. 하나씩 눌러 봐.`, `${n}곳 정도 갈림길이 보여.`]],
      clean: ['happy', ['실수가 안 보여. 정말 잘 뒀다.', '깔끔했어. 배울 게 많네.']],
      none: ['idle', ['확실하게 짚을 곳은 없어. 조금씩 밀렸을 뿐이야.', '여기다 할 실수는 못 찾았어. 다음엔 될 거야.']],
      missedWin: (c, n) => ['surprised', n === 1
        ? [`어라, ${c}에 두면 바로 다섯이었네.`, `여기 ${c}가 비어 있었어.`]
        : [`어라, 여기 ${c}부터 넷을 이으면 이겼어.`, `${c}부터 계속 넷을 만들면 됐었네.`]],
      turn: (c) => ['think', [`여기서 흐름이 나한테 왔어. ${c}면 아직 비슷했을 거야.`, `이 수가 갈림길이었어. ${c}에 두면 버틸 만했어.`]],
    },
    puzzle: {
      intro: (n) => ['idle', [`넷을 계속 만들어서 ${n}수 안에 이겨 봐.`, `넷으로만 몰아서 ${n}수 만에 끝내 보자.`]],
      notFour: ['idle', ['음, 넷이 아니네. 넷으로만 이어 가 봐.', '그건 넷이 아니야. 다시 골라 봐.']],
      blocked: (c) => ['nervous', [`${c} 막았어. 다음은?`, `${c}에 둘게. 아직 길이 있을까?`]],
      trapped: ['surprised', ['어라, 막을 자리가 금수네. 못 막겠다.', '거긴 금수라 못 막아. 잘 찾았다.']],
      solved: ['happy', ['풀었네. 그 길 정말 좋다.', '정답이야. 방금 수순 기억해 둘래.']],
      failed: ['idle', ['음, 이제 넷이 안 이어지네. 다시 해 보자.', '길이 끊겼어. 처음부터 천천히.']],
      hint: (c) => ['happy', [`${c}부터 시작해 봐.`, `첫 수는 ${c}야. 다음은 직접 찾아 봐.`]],
    },
  },
};

const pick = (options: string[], seed: number) => options[Math.abs(seed) % options.length];
const at = (c: Coord | undefined) => (c ? formatCoord(c.row, c.col) : '');
const isFour = (t: BoardThreat) => t.type === 'OPEN_FOUR' || t.type === 'CLOSED_FOUR';

export function catLines(ctx: CatContext, voice: CatVoice = VOICES.kkamang): CatLine[] {
  const human: Player = ctx.aiPlayer === BLACK ? 2 : 1;
  const seed = ctx.moveCount + (ctx.lastMove ? ctx.lastMove.row * 17 + ctx.lastMove.col * 7 : 0);
  const out: CatLine[] = [];
  const add = (key: string, [mood, options]: Say, tone: CatLine['tone'] = 'default') =>
    out.push({ key, mood, tone, text: pick(options, seed) });

  if (ctx.winner) {
    add('end', ctx.winner.winner === ctx.aiPlayer ? voice.win : voice.lose);
    return out;
  }
  if (ctx.isDraw) {
    add('end', voice.draw);
    return out;
  }

  const vcf = ctx.activeVcf;
  if (vcf && vcf.length > 0) {
    const c = at(vcf[0]);
    const n = vcf.length;
    if (ctx.turnPlayer === ctx.aiPlayer) add(`ai-vcf:${c}`, n <= 1 ? voice.aiVcfNow(c) : voice.aiVcf(c, n));
    else add(`human-vcf:${c}`, n <= 1 ? voice.humanVcfNow(c) : voice.humanVcf(c, n));
  }

  for (const t of ctx.threats.filter(isFour)) {
    const c = at(t.targetCells[0]);
    if (t.player === ctx.aiPlayer && ctx.turnPlayer === human) add(`ai-four:${c}`, voice.aiFour(c), 'warn');
    else if (t.player === human && ctx.turnPlayer === ctx.aiPlayer) add(`human-four:${c}`, voice.humanFour(c));
  }

  if (ctx.aiPendingVcf && ctx.aiPendingVcf.length > 0 && ctx.turnPlayer === human) {
    const c = at(ctx.aiPendingVcf[0]);
    add(`ai-threat:${c}`, voice.aiThreat(c), 'warn');
  }

  const last = ctx.lastMove;
  if (ctx.moveCount === 1 && last && last.player === human) {
    add('first-move', last.row === 7 && last.col === 7 ? voice.firstCenter : voice.firstOff);
  }

  if (last?.player === human && last.tag === '4·3') add(`fork:${ctx.moveCount}`, voice.fork);

  // Swings over one full move (same side to move), so tempo alone never counts.
  const aiSign = ctx.aiPlayer === BLACK ? 1 : -1;
  if (typeof ctx.prevEvalScore === 'number') {
    const aiDelta = aiSign * (ctx.evalScore - ctx.prevEvalScore);
    if (aiDelta <= -0.25) {
      add(`swing:${ctx.moveCount}`, last?.player === ctx.aiPlayer ? voice.blunder : voice.humanGood);
    } else if (aiDelta >= 0.25 && last?.player === ctx.aiPlayer) {
      add(`swing:${ctx.moveCount}`, voice.aiGood);
    }
  }

  const aiBias = aiSign * ctx.evalScore;
  const pliesLeft = ctx.nnPredictedPliesLeft;
  if (voice.closing && aiBias >= 0.55 && pliesLeft != null && pliesLeft <= 16) {
    add('closing', voice.closing(Math.max(2, Math.round(pliesLeft / 2))));
  } else if (voice.losing && aiBias <= -0.55 && pliesLeft != null && pliesLeft <= 14) {
    add('losing', voice.losing(Math.max(2, Math.round(pliesLeft / 2))));
  }

  for (const t of ctx.threats.filter((x) => x.type === 'OPEN_THREE')) {
    const id = t.stones.map(at).sort().join('-');
    if (t.player === ctx.aiPlayer && ctx.turnPlayer === human) add(`ai-three:${id}`, voice.aiThree(at(t.targetCells[0])));
    else if (t.player === human) add(`human-three:${id}`, voice.humanThree);
  }

  if (aiBias >= 0.3) add('ahead', voice.ahead);
  else if (aiBias <= -0.3) add('behind', voice.behind);

  return out;
}

/** Expression when there is nothing to say. */
export function catQuietMood(ctx: CatContext): CatMood {
  const aiBias = (ctx.aiPlayer === BLACK ? 1 : -1) * ctx.evalScore;
  return aiBias >= 0.3 ? 'smug' : aiBias <= -0.3 ? 'nervous' : 'idle';
}

const FORBIDDEN_NAME = { THREE_THREE: '3·3', FOUR_FOUR: '4·4', OVERLINE: '장목' } as const;

export function catForbiddenLine(
  coord: string,
  reason: keyof typeof FORBIDDEN_NAME,
  seed: number,
  voice: CatVoice = VOICES.kkamang,
): string {
  const name = FORBIDDEN_NAME[reason];
  const because = reason === 'FOUR_FOUR' ? `${name}라서` : `${name}이라서`;
  return pick(voice.forbidden(coord, because, name), seed);
}

export const pickLine = pick;
