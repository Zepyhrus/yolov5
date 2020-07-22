


from hie.hie import HIE
from hie.tools import GREEN


if __name__ == "__main__":
  dt = HIE('res.json', 'seed')

  dt.viz(show_bbox=True, color=GREEN, pause=5)

