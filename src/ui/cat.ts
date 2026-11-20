// The cat: 12 sprite poses, a speech bubble that belongs to it, and idle blinking.
// Swapping the art later (pixel art, GIF) only means replacing files in src/assets/<cat dir>.

/** 까망이 (src/assets/cat) and 하양이 (src/assets/white). */
export type CatId = 'kkamang' | 'hayang';
const DIR: Record<CatId, string> = { kkamang: 'cat', hayang: 'white' };

export type CatMood =
  | 'idle'
  | 'happy'
  | 'think'
  | 'smug'
  | 'surprised'
  | 'nervous'
  | 'sulky'
  | 'warn'
  | 'win'
  | 'lose'
  | 'sleepy';

const SPRITES = import.meta.glob('../assets/{cat,white}/*.png', {
  eager: true,
  query: '?url',
  import: 'default',
}) as Record<string, string>;

export function catSpriteUrl(pose: CatMood | 'idle-blink', cat: CatId = 'kkamang'): string {
  return SPRITES[`../assets/${DIR[cat]}/${pose}.png`];
}

// Preload so pose changes never flash empty.
for (const url of Object.values(SPRITES)) new Image().src = url;

export class CatView {
  readonly el = document.createElement('div');
  private bubble = document.createElement('p');
  private figure = document.createElement('button');
  private layers = [document.createElement('img'), document.createElement('img')];
  private front = 0;
  private mood: CatMood = 'idle';
  private id: CatId = 'kkamang';

  constructor(onPoke: () => void) {
    this.el.className = 'cat';
    this.el.dataset.mood = 'idle';
    this.bubble.className = 'cat-bubble';
    this.bubble.setAttribute('aria-live', 'polite');
    this.bubble.hidden = true;
    this.figure.type = 'button';
    this.figure.className = 'cat-figure';
    this.figure.setAttribute('aria-label', '까망이 쓰다듬기');
    this.figure.addEventListener('click', onPoke);
    for (const img of this.layers) {
      img.alt = '';
      img.draggable = false;
      this.figure.appendChild(img);
    }
    this.layers[0].src = catSpriteUrl('idle');
    this.layers[0].classList.add('on');
    this.el.append(this.bubble, this.figure);
    this.scheduleBlink();
  }

  setCharacter(id: CatId, label: string): void {
    if (id === this.id) return;
    this.id = id;
    this.el.dataset.cat = id;
    this.figure.setAttribute('aria-label', `${label} 쓰다듬기`);
    this.layers[this.front].src = catSpriteUrl(this.mood, id);
  }

  setMood(mood: CatMood): void {
    if (mood === this.mood) return;
    this.mood = mood;
    this.el.dataset.mood = mood;
    this.show(mood);
    this.figure.classList.remove('bounce');
    void this.figure.offsetWidth; // restart the bounce
    this.figure.classList.add('bounce');
  }

  /** Show a line, or hide the bubble with null. */
  say(text: string | null, tone: 'default' | 'warn' = 'default'): void {
    if (!text) {
      this.bubble.hidden = true;
      return;
    }
    const changed = this.bubble.hidden || this.bubble.textContent !== text;
    this.bubble.hidden = false;
    this.bubble.textContent = text;
    this.bubble.classList.toggle('warn', tone === 'warn');
    if (changed) {
      this.bubble.classList.remove('pop');
      void this.bubble.offsetWidth;
      this.bubble.classList.add('pop');
    }
  }

  /** Cross-fade between two stacked images. */
  private show(pose: CatMood | 'idle-blink'): void {
    const next = 1 - this.front;
    this.layers[next].src = catSpriteUrl(pose, this.id);
    this.layers[next].classList.add('on');
    this.layers[this.front].classList.remove('on');
    this.front = next;
  }

  private scheduleBlink(): void {
    window.setTimeout(() => {
      if (this.mood === 'idle' && this.el.isConnected) {
        this.layers[this.front].src = catSpriteUrl('idle-blink', this.id);
        window.setTimeout(() => {
          if (this.mood === 'idle') this.layers[this.front].src = catSpriteUrl('idle', this.id);
        }, 160);
      }
      this.scheduleBlink();
    }, 2800 + Math.random() * 2600);
  }
}
