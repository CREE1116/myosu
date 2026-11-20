// 새 소식: newest first. Add an entry at the top for each release; a new id lights the lobby badge.

export type PatchTag = '모델' | '기능' | '묘수풀이' | '개선';

export interface PatchNote {
  id: string;
  date: string;
  title: string;
  items: Array<{ tag: PatchTag; text: string }>;
}

export const PATCH_NOTES: PatchNote[] = [
  {
    id: '2026-11-21-b',
    date: '2026.11.21',
    title: '복기 승부처 · 묘수풀이 · 더 강해진 고양이',
    items: [
      { tag: '모델', text: '까망이와 하양이의 두뇌를 gen 314로 바꿨어요. 학습 방식을 고쳐서 다시 꾸준히 강해지는 중이에요.' },
      { tag: '기능', text: '기보실에서 판을 열면 고양이가 승부처를 짚어 줘요. 놓친 필승과 판이 넘어간 수만, 확인된 것만 알려 줘요.' },
      { tag: '기능', text: '복기 중 내 차례인 수에서 "다시 두기"를 누르면 그 자리부터 같은 고양이와 다시 겨룰 수 있어요.' },
      { tag: '묘수풀이', text: '묘수풀이 30문제가 생겼어요. 넷 이어 가기 12, 금수 함정 6, 긴 수순 12문제예요.' },
      { tag: '묘수풀이', text: '복기에서 놓친 필승은 "내 기보에서" 테마에 문제로 모여요.' },
      { tag: '개선', text: '휴대폰 복기 화면을 정리했어요. 고양이 말과 판, 승부처가 한 화면에 보여요.' },
      { tag: '개선', text: '휴대폰에서 화면이 밀려 엉뚱한 곳에 두어지던 문제를 고쳤어요. 이제 한 번 탭하면 바로 둬요. 두 번 탭해서 두고 싶으면 "조준 착수"를 켜세요.' },
    ],
  },
  {
    id: '2026-11-21-a',
    date: '2026.11.21',
    title: '첫 공개: 까망이와 하양이',
    items: [
      { tag: '기능', text: '오목을 좋아하는 까망이(보통)와 느긋한 하양이(쉬움) 중에 골라 둘 수 있어요.' },
      { tag: '기능', text: '막히면 다른 고양이가 와서 한 수 추천이나 필승 수를 한 번씩 알려 줘요. 무르기는 까망이 1번, 하양이 3번이에요.' },
      { tag: '개선', text: '휴대폰에서 한 화면에 판이 다 보이고, 홈 화면에 설치하면 오프라인에서도 둘 수 있어요.' },
    ],
  },
];

const SEEN_KEY = 'myosu_news_seen';

export function hasUnseenNews(): boolean {
  try {
    return localStorage.getItem(SEEN_KEY) !== PATCH_NOTES[0].id;
  } catch {
    return false;
  }
}

export function markNewsSeen(): void {
  try {
    localStorage.setItem(SEEN_KEY, PATCH_NOTES[0].id);
  } catch {
    // Blocked storage: the badge just shows again next time.
  }
}
