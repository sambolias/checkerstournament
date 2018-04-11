#include "NeuralNetworkPlayer.h"
#include "NeuralNetworkFileHandler.h"
using std::shared_ptr;
using std::string;
#include <iostream>
using std::cout;

NeuralNetworkPlayer::NeuralNetworkPlayer(shared_ptr<NeuralNetwork> network, string name, char color): ComputerPlayer(color)
{
	_neuralNetwork = network;
	SetName(name);
	Reset();
}

shared_ptr<NeuralNetworkPlayer> NeuralNetworkPlayer::clone()
{
	return std::make_shared<NeuralNetworkPlayer>(_neuralNetwork, _name, _color);
}

void NeuralNetworkPlayer::saveNetwork()
{
	NeuralNetworkFileHandler n;
	n.WriteNetworkToFile("winner.txt", _neuralNetwork);
}

double NeuralNetworkPlayer::getHeuristic(Board& board)
{
	bool isRedPlayer = (_color == Board::RED);
	return _neuralNetwork->GetBoardEvaluation(isRedPlayer, board.GetBoard());
}

shared_ptr<NeuralNetwork> NeuralNetworkPlayer::GetNeuralNetork()
{
	return _neuralNetwork;
}

const string & NeuralNetworkPlayer::GetName()
{
	return _name;
}

double NeuralNetworkPlayer::GetScore() const
{
	return _score;
}

double NeuralNetworkPlayer::GetScore()
{
	return _score;
}

int NeuralNetworkPlayer::GetGamesPlayed() const
{
	return _gamesPlayed;
}

int NeuralNetworkPlayer::GetGamesPlayed()
{
	return _gamesPlayed;
}

void NeuralNetworkPlayer::AddScore(double value)
{
	_score += value;
}

void NeuralNetworkPlayer::SetColor(char color)
{
	_color = color;
}

void NeuralNetworkPlayer::IncrementGamesPlayed()
{
	_gamesPlayed++;
}

void NeuralNetworkPlayer::Reset()
{
	_score = 0;
	_gamesPlayed = 0;
}

void NeuralNetworkPlayer::SetName(string name)
{
	_name = name;
}
