#include "sd_configuration.h"
#include "configuration.h"
#include <string>

namespace strength_detection {

using namespace minizero;

int players_per_batch = 18;
int games_per_player = 9;
int n_frames = 50;
int move_step_to_choose = 4;

std::string sgf_file_path = "../../data_set/";

void setConfiguration(config::ConfigureLoader& cl)
{
    config::setConfiguration(cl);

    cl.addParameter("players_per_batch", players_per_batch, "", "Strength Detection");
    cl.addParameter("games_per_player", games_per_player, "", "Strength Detection");
    cl.addParameter("n_frames", n_frames, "", "Strength Detection");
    cl.addParameter("move_step_to_choose", move_step_to_choose, "", "Strength Detection");
    cl.addParameter("sgf_file_path", sgf_file_path, "", "Strength Detection");
}

} // namespace strength_detection
