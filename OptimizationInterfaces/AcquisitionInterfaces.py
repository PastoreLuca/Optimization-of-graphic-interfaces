import os
import time
from uiautomator import Device

def get_screenshot_and_xml(d, count):
    os.makedirs(f'screenshotsAndSnapshots', exist_ok=True)
    filename = f'screenshotsAndSnapshots/gui_{count}'
    d.screenshot(f'{filename}.png')
    xml = d.dump()
    with open(f'{filename}.xml', 'w') as file:
        file.write(xml)
    return xml

# Rimuovi app_name dalla lista di parametri
def navigate_interfaces(d, visited, clicked, main_interface, depth=0, count=0):
    xml = get_screenshot_and_xml(d, count)
    if xml in visited:
        return False
    visited.add(xml)
    elements = d(clickable=True)
    for i, element in enumerate(elements):
        element_id = get_element_id(element)
        if element_id in clicked or not is_navigation_element(element):
            continue
        try:
            element.click.wait()
            time.sleep(1)
            new_xml = d.dump()
            if new_xml == xml:  # Se l'interfaccia non è cambiata, ignora questo elemento
                continue
            count += 1
            clicked.add(element_id)
            if navigate_interfaces(d, visited, clicked, main_interface, depth, count):
                return True
            # Verifica se l'interfaccia corrente è l'interfaccia principale
            if new_xml != main_interface:
                d.press.back()
        except Exception as e:
            print(f"Cannot click on element {i}: {e}")

def iterative_deepening(d, visited, clicked, max_depth=10):
    main_interface = d.dump()  # Salva l'interfaccia principale
    for depth in range(max_depth):
        if navigate_interfaces(d, visited, clicked, main_interface, depth):
            break

def main():
    d = Device('39V4C19320009558')  # Sostituisci con l'ID del tuo dispositivo
    visited = set()
    clicked = set()
    iterative_deepening(d, visited, clicked)

if __name__ == "__main__":
    main()
