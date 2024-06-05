from package.src.pedestrians_social_binding.environment import Environment
from package.src.pedestrians_social_binding.threshold import Threshold
from package.src.pedestrians_social_binding.plot_utils import *
from package.src.pedestrians_social_binding.constants import *

if __name__ == "__main__":
    env_values = {
        "atc": SOCIAL_RELATIONS_EN,
        "diamor": INTENSITIES_OF_INTERACTION_NUM,
    }
    env_bonding = {"atc": "soc_rel", "diamor": "interaction"}

    for env_name in ["atc", "diamor"]:
        print(f"Environment: {env_name}")
        env = Environment(env_name, data_dir="../data/formatted")

        for day in env.get_days():
            print(f" - Day: {day}")
            groups = env.get_groups_grouped_by(
                env_bonding[env_name], size=2, days=[day]
            )

            total = 0
            for soc_rel in groups.keys():
                if env_name == "atc" and soc_rel > 0 or env_name == "diamor":
                    print(f"  {env_values[env_name][soc_rel]}: {len(groups[soc_rel])}")
                    total += len(groups[soc_rel])

            print(f"  Total: {total}")
