import planetTexture from "./planet-texture.svg";
import "./SpaceBackground.css";

function makeStars() {
  let seed = 4271;

  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  return Array.from({ length: 30 }, (_, id) => ({
    id,
    left: `${random() * 100}%`,
    top: `${random() * 100}%`,
    size: `${1 + random() * 1.5}px`,
    duration: `${3.5 + random() * 6}s`,
    delay: `${-random() * 12}s`,
    brightness: 0.7 + random() * 0.3,
  }));
}

const STARS = makeStars();

function OrbitLayer({ side }) {
  return (
    <div className={`sky-orbit-layer ${side}`}>
      {["one", "two"].map((orbit) => (
        <div
          className={`sky-orbit-path ${orbit}`}
          key={orbit}
        >
          <div className="sky-orbit-spinner">
            <span className="sky-satellite">
              <span>{orbit === "one" ? "✦" : "✧"}</span>
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function SpaceBackground() {
  return (
    <div className="sky-background" aria-hidden="true">
      {STARS.map((star) => (
        <span
          key={star.id}
          className="sky-star"
          style={{
            left: star.left,
            top: star.top,
            "--star-size": star.size,
            "--star-duration": star.duration,
            "--star-delay": star.delay,
            "--star-brightness": star.brightness,
          }}
        />
      ))}

      <div className="sky-planet">
        <OrbitLayer side="back" />

        <div className="sky-globe">
          <div
            className="sky-surface"
            style={{
              backgroundImage: `url("${planetTexture}")`,
            }}
          />
        </div>

        <OrbitLayer side="front" />
      </div>
    </div>
  );
}