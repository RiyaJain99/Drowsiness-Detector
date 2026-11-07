/**
 * The road behind the cockpit. Its speed comes from `--road-speed`, set per
 * driver state in index.css: cruising when alert, slowing when drowsy, stopped
 * at critical. Decorative, so it stays out of the accessibility tree.
 */
export function Backdrop() {
  return (
    <div className="backdrop" aria-hidden="true">
      <div className="backdrop-glow one" />
      <div className="backdrop-glow two" />
      <div className="road">
        <div className="road-plane">
          <div className="road-grid" />
          <div className="road-lane" />
        </div>
      </div>
    </div>
  );
}
